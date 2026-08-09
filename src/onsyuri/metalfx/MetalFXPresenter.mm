// Objective-C++ implementation of the MetalFX Spatial presenter.
// Compiled only on macOS (config_darwin in CMakeLists.txt).

#include "MetalFXPresenter.h"

#include "cunny/cunny_metal.h"

#include <Metal/Metal.h>
#include <MetalFX/MetalFX.h>
#include <QuartzCore/CAMetalLayer.h>
#include <SDL.h>
#include <SDL_metal.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <memory>
#include <vector>

namespace onsyuri_metalfx {

namespace {

const char *kPixelFormatBGRA8 = "BGRA8Unorm";

// Keep this in sync with the pixel layout of SDL_PIXELFORMAT_ARGB8888:
// little-endian memory order is B, G, R, A, which matches MTLPixelFormatBGRA8Unorm.
const MTLPixelFormat kFramePixelFormat = MTLPixelFormatBGRA8Unorm;

void logLine(const char *fmt, ...) {
    char buf[512];
    va_list args;
    va_start(args, fmt);
    vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);
    // Route through the existing ONS logging so it lands in the same
    // stdout/stderr stream Yoghourt captures.
    fputs(buf, stdout);
    fputc('\n', stdout);
    fflush(stdout);
}

} // namespace

struct Impl {
    SDL_Window *window;
    id<MTLDevice> device;
    CAMetalLayer *layer;
    id<MTLCommandQueue> queue;
    id<MTLFXSpatialScaler> scaler;
    id<MTLFXSpatialScaler> cunnyScaler;
    id<MTLTexture> inputTexture;
    id<MTLTexture> outputTexture;
    id<MTLTexture> maskTexture;
    id<MTLTexture> aaTexture;
    id<MTLRenderPipelineState> pipeline;
    id<MTLRenderPipelineState> maskedPipeline;
    id<MTLRenderPipelineState> aaPipeline;
    id<MTLSamplerState> sampler;
    MTLTextureUsage colorUsage;
    MTLTextureUsage outputUsage;
    MTLFXSpatialScalerColorProcessingMode colorMode;

    SDL_Surface *textLayer = nullptr;
    bool maskEnabled = true;
    bool testMask = false;
    bool aaEnabled = false;
    bool aaDisabledLogged = false;
    char *maskDumpPath = nullptr;
    bool maskDumped = false;

    // Scaler selection: 0 = MetalFX (default), 1 = CuNNy only, 2 = CuNNy + MetalFX.
    int scalerMode = 0;
    bool cunnyReady = false;
    id<MTLComputePipelineState> cunnyP1;
    id<MTLComputePipelineState> cunnyP2;
    id<MTLComputePipelineState> cunnyP3;
    id<MTLComputePipelineState> cunnyP4;
    id<MTLComputePipelineState> downscalePS;
    id<MTLTexture> cunnyOut;        // 2x BGRA8Unorm output of the CNN
    id<MTLTexture> cunnyT0, cunnyT1, cunnyT2, cunnyT3; // R8G8B8A8_UNORM intermediates
    id<MTLTexture> downscaleOut;    // target-sized output when target < 2x
    char *cunnyDumpDir = nullptr;
    bool cunnyDumped = false;

    int gameW = 0;
    int gameH = 0;
    int drawW = 0;
    int drawH = 0;
    int outW = 0;
    int outH = 0;
    int outX = 0;
    int outY = 0;
    bool useScaler = false;
    bool active = false;
    // Completion handlers outlive individual present() calls and may run after
    // Presenter destruction. Keep only this shared failure signal in them.
    // 0 = none, 1 = retry without CuNNy, 2 = disable the Metal presenter.
    std::shared_ptr<std::atomic<int>> asyncFailure = std::make_shared<std::atomic<int>>(0);
    bool diagnosticsLogged = false;
    int frameCount = 0;
    CFAbsoluteTime lastFreqLog = 0;
    int freqFrames = 0;
};

bool rebuildForOutputSize(Impl &p);
bool ensurePipeline(Impl &p);
bool encodeAAPass(id<MTLCommandBuffer> cb, Impl &p, id<MTLTexture> src, id<MTLTexture> dst,
                  id<MTLTexture> mask, id<MTLRenderPipelineState> pipeline,
                  float texelX, float texelY);
bool ensureCunny(Impl &p);
bool encodeCunny(id<MTLCommandBuffer> cb, Impl &p, id<MTLTexture> input);
bool encodeDownscale(id<MTLCommandBuffer> cb, Impl &p, id<MTLTexture> src, id<MTLTexture> dst);

Presenter::Presenter(SDL_Window *window, void *metal_view, int game_width, int game_height)
    : impl_(new Impl) {
    impl_->window = window;
    impl_->gameW = game_width;
    impl_->gameH = game_height;

    const char *mask_env = getenv("YOGHOURT_ONS_METALFX_TEXTMASK");
    if (mask_env && strcmp(mask_env, "0") == 0) {
        impl_->maskEnabled = false;
    }
    const char *dump_env = getenv("YOGHOURT_ONS_METALFX_MASKDUMP");
    if (dump_env && dump_env[0]) {
        impl_->maskDumpPath = strdup(dump_env);
    }
    const char *test_env = getenv("YOGHOURT_ONS_METALFX_TESTMASK");
    if (test_env && strcmp(test_env, "1") == 0) {
        impl_->testMask = true;
    }
    // Selective AA: masked FXAA at game resolution before MetalFX.
    const char *aa_env = getenv("YOGHOURT_ONS_METALFX_AA");
    if (aa_env && strcmp(aa_env, "1") == 0) {
        impl_->aaEnabled = true;
    }
    // Scaler selection: metalfx (default) | cunny | cunny+metalfx.
    const char *scaler_env = getenv("YOGHOURT_ONS_METALFX_SCALER");
    if (scaler_env) {
        if (strcmp(scaler_env, "cunny") == 0) {
            impl_->scalerMode = 1;
        } else if (strcmp(scaler_env, "cunny+metalfx") == 0) {
            impl_->scalerMode = 2;
        }
    }
    const char *cunny_dump_env = getenv("YOGHOURT_ONS_METALFX_CUNNY_DUMP");
    if (cunny_dump_env && cunny_dump_env[0]) {
        impl_->cunnyDumpDir = strdup(cunny_dump_env);
    }

    if (!metal_view) {
        logLine("[MetalFX] disabled: no metal view");
        return;
    }
    CAMetalLayer * layer = (__bridge CAMetalLayer *)SDL_Metal_GetLayer((SDL_MetalView)metal_view);
    if (!layer) {
        logLine("[MetalFX] disabled: SDL_Metal_GetLayer failed");
        return;
    }
    impl_->layer = layer;

    id<MTLDevice> device = layer.device;
    if (!device) {
        device = MTLCreateSystemDefaultDevice();
        layer.device = device;
    }
    if (!device) {
        logLine("[MetalFX] disabled: no Metal device");
        return;
    }
    impl_->device = device;

    if (![MTLFXSpatialScalerDescriptor supportsDevice:device]) {
        logLine("[MetalFX] disabled: MTLFXSpatialScaler not supported on this device");
        return;
    }

    id<MTLCommandQueue> queue = [device newCommandQueue];
    if (!queue) {
        logLine("[MetalFX] disabled: command queue creation failed");
        return;
    }
    impl_->queue = queue;

    int dw = 0, dh = 0;
    SDL_Metal_GetDrawableSize(window, &dw, &dh);
    if (dw <= 0 || dh <= 0) {
        logLine("[MetalFX] disabled: invalid initial drawable size %dx%d", dw, dh);
        return;
    }
    impl_->drawW = dw;
    impl_->drawH = dh;

    impl_->colorMode = MTLFXSpatialScalerColorProcessingModePerceptual;
    if (!rebuildForOutputSize(*impl_)) {
        return; // rebuild already logged the reason
    }

    if (impl_->scalerMode != 0 && !ensureCunny(*impl_)) {
        // CuNNy unavailable: fall back to the MetalFX path for this session.
        impl_->scalerMode = 0;
    }

    impl_->active = true;
}

Presenter::~Presenter() {
    if (impl_->maskDumpPath) {
        free(impl_->maskDumpPath);
    }
    if (impl_->cunnyDumpDir) {
        free(impl_->cunnyDumpDir);
    }
    delete impl_;
}

bool Presenter::isActive() const {
    return impl_->active;
}

void Presenter::setTextLayer(SDL_Surface *textLayer) {
    impl_->textLayer = textLayer;
}// Recomputes the letterboxed content area for the current drawable size and
// (re)creates scaler + output texture when the content size changed.
// Returns false and logs when anything fails (session must be disabled).
bool rebuildForOutputSize(Impl &p) {
    const float scale = std::min((float)p.drawW / (float)p.gameW,
                                 (float)p.drawH / (float)p.gameH);
    const int outW = std::max(1, (int)std::lroundf(p.gameW * scale));
    const int outH = std::max(1, (int)std::lroundf(p.gameH * scale));

    if (outW == p.outW && outH == p.outH) {
        // Content area unchanged; only the letterbox offset may have moved.
        p.outX = (p.drawW - outW) / 2;
        p.outY = (p.drawH - outH) / 2;
        return true;
    }

    // Only upscale with MetalFX when the content area is actually larger than
    // the game frame; otherwise the final pass samples the input directly.
    logLine("[MetalFX] rebuild: output %dx%d -> %dx%d (drawable %dx%d)", p.outW, p.outH, outW, outH, p.drawW, p.drawH);
    p.useScaler = (outW > p.gameW || outH > p.gameH);
    p.outW = outW;
    p.outH = outH;
    p.outX = (p.drawW - outW) / 2;
    p.outY = (p.drawH - outH) / 2;

    p.scaler = nil;
    p.cunnyScaler = nil;
    p.outputTexture = nil;

    if (p.useScaler) {
        MTLFXSpatialScalerDescriptor *desc = [[MTLFXSpatialScalerDescriptor alloc] init];
        desc.colorTextureFormat = kFramePixelFormat;
        desc.outputTextureFormat = kFramePixelFormat;
        desc.inputWidth = (NSUInteger)p.gameW;
        desc.inputHeight = (NSUInteger)p.gameH;
        desc.outputWidth = (NSUInteger)outW;
        desc.outputHeight = (NSUInteger)outH;
        desc.colorProcessingMode = p.colorMode;
        p.scaler = [desc newSpatialScalerWithDevice:p.device];
        if (!p.scaler) {
            logLine("[MetalFX] disabled: scaler creation failed at output %dx%d", outW, outH);
            p.active = false;
            return false;
        }
        p.colorUsage = p.scaler.colorTextureUsage;
        p.outputUsage = p.scaler.outputTextureUsage;

        // The combined path feeds MetalFX with CuNNy's fixed 2x output. MetalFX
        // requires the color texture dimensions to match the descriptor, so it
        // needs a separate scaler from the original-resolution fallback path.
        const int c2w = p.gameW * 2;
        const int c2h = p.gameH * 2;
        if (p.scalerMode == 2 && (outW > c2w || outH > c2h)) {
            MTLFXSpatialScalerDescriptor *cunnyDesc = [[MTLFXSpatialScalerDescriptor alloc] init];
            cunnyDesc.colorTextureFormat = kFramePixelFormat;
            cunnyDesc.outputTextureFormat = kFramePixelFormat;
            cunnyDesc.inputWidth = (NSUInteger)c2w;
            cunnyDesc.inputHeight = (NSUInteger)c2h;
            cunnyDesc.outputWidth = (NSUInteger)outW;
            cunnyDesc.outputHeight = (NSUInteger)outH;
            cunnyDesc.colorProcessingMode = p.colorMode;
            p.cunnyScaler = [cunnyDesc newSpatialScalerWithDevice:p.device];
            if (!p.cunnyScaler) {
                logLine("[MetalFX] cunny+metalfx unavailable at output %dx%d; using CuNNy only", outW, outH);
                p.scalerMode = 1;
            } else {
                p.outputUsage |= p.cunnyScaler.outputTextureUsage;
                const MTLTextureUsage requiredCunnyUsage =
                    MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite | p.cunnyScaler.colorTextureUsage;
                if (p.cunnyOut && (p.cunnyOut.usage & requiredCunnyUsage) != requiredCunnyUsage) {
                    MTLTextureDescriptor *cunnyOutDesc =
                        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:kFramePixelFormat
                                                                          width:(NSUInteger)c2w
                                                                         height:(NSUInteger)c2h
                                                                      mipmapped:NO];
                    cunnyOutDesc.storageMode = MTLStorageModePrivate;
                    cunnyOutDesc.usage = requiredCunnyUsage;
                    id<MTLTexture> replacement = [p.device newTextureWithDescriptor:cunnyOutDesc];
                    if (!replacement) {
                        logLine("[MetalFX] cunny+metalfx output texture recreation failed; using CuNNy only");
                        p.cunnyScaler = nil;
                        p.scalerMode = 1;
                    } else {
                        p.cunnyOut = replacement;
                    }
                }
            }
        }
    } else {
        p.colorUsage = MTLTextureUsageShaderRead;
        p.outputUsage = MTLTextureUsageShaderWrite | MTLTextureUsageShaderRead;
    }

    if (!p.inputTexture) {
        MTLTextureDescriptor *inDesc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:kFramePixelFormat
                                                                                          width:(NSUInteger)p.gameW
                                                                                         height:(NSUInteger)p.gameH
                                                                                      mipmapped:NO];
        inDesc.storageMode = MTLStorageModePrivate;
        inDesc.usage = p.colorUsage;
        p.inputTexture = [p.device newTextureWithDescriptor:inDesc];
        if (!p.inputTexture) {
            logLine("[MetalFX] disabled: input texture creation failed");
            p.active = false;
            return false;
        }
    }

    MTLTextureDescriptor *outDesc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:kFramePixelFormat
                                                                                       width:(NSUInteger)outW
                                                                                      height:(NSUInteger)outH
                                                                                   mipmapped:NO];
    outDesc.storageMode = MTLStorageModePrivate;
    outDesc.usage = p.outputUsage;
    p.outputTexture = [p.device newTextureWithDescriptor:outDesc];
    if (!p.outputTexture) {
        logLine("[MetalFX] disabled: output texture creation failed at %dx%d", outW, outH);
        p.active = false;
        return false;
    }

    if (p.maskEnabled && !p.maskTexture) {
        MTLTextureDescriptor *maskDesc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:kFramePixelFormat
                                                                                            width:(NSUInteger)p.gameW
                                                                                           height:(NSUInteger)p.gameH
                                                                                        mipmapped:NO];
        maskDesc.storageMode = MTLStorageModePrivate;
        maskDesc.usage = MTLTextureUsageShaderRead;
        p.maskTexture = [p.device newTextureWithDescriptor:maskDesc];
        if (!p.maskTexture) {
            logLine("[MetalFX] disabled: mask texture creation failed");
            p.active = false;
            return false;
        }
    }

    if (p.aaEnabled && !p.aaTexture) {
        MTLTextureDescriptor *aaDesc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:kFramePixelFormat
                                                                                          width:(NSUInteger)p.gameW
                                                                                         height:(NSUInteger)p.gameH
                                                                                      mipmapped:NO];
        aaDesc.storageMode = MTLStorageModePrivate;
        aaDesc.usage = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget | p.colorUsage;
        p.aaTexture = [p.device newTextureWithDescriptor:aaDesc];
        if (!p.aaTexture) {
            logLine("[MetalFX] disabled: aa texture creation failed");
            p.active = false;
            return false;
        }
    }

    // Target-sized downscale destination for the CuNNy path (target < 2x).
    if (p.scalerMode != 0 && (p.downscaleOut == nil || p.downscaleOut.width != (NSUInteger)outW ||
                              p.downscaleOut.height != (NSUInteger)outH)) {
        MTLTextureDescriptor *dsDesc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:kFramePixelFormat
                                                                                         width:(NSUInteger)outW
                                                                                        height:(NSUInteger)outH
                                                                                     mipmapped:NO];
        dsDesc.storageMode = MTLStorageModePrivate;
        dsDesc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
        p.downscaleOut = [p.device newTextureWithDescriptor:dsDesc];
        if (!p.downscaleOut) {
            logLine("[MetalFX] cunny disabled: downscale texture creation failed at %dx%d", outW, outH);
            p.scalerMode = 0;
            return true;
        }
    }

    return true;
}

// Creates the minimal final-pass pipeline (fullscreen quad + one sampler).
bool ensurePipeline(Impl &p) {
    if (p.pipeline) {
        return true;
    }

    static const char *kMSL =
        "#include <metal_stdlib>\n"
        "using namespace metal;\n"
        "struct VOut { float4 pos [[position]]; float2 uv; };\n"
        "vertex VOut vs(uint vid [[vertex_id]],\n"
        "               constant float2 *pos [[buffer(0)]],\n"
        "               constant float2 *uvs [[buffer(1)]]) {\n"
        "  VOut o; o.pos = float4(pos[vid], 0.0, 1.0); o.uv = uvs[vid]; return o;\n"
        "}\n"
        "fragment float4 fs(VOut in [[stage_in]],\n"
        "                   texture2d<float> tex [[texture(0)]],\n"
        "                   sampler smp [[sampler(0)]]) {\n"
        "  return tex.sample(smp, in.uv);\n"
        "}\n"
        // Text-protected final pass: masked regions (glyph alpha) show the
        // plain linear-upscaled frame so MetalFX reconstruction never
        // touches text; unmasked regions show the MetalFX result.
        "fragment float4 fsMasked(VOut in [[stage_in]],\n"
        "                         texture2d<float> tex [[texture(0)]],\n"
        "                         texture2d<float> plain [[texture(1)]],\n"
        "                         texture2d<float> mask [[texture(2)]],\n"
        "                         sampler smp [[sampler(0)]]) {\n"
        "  float4 m = mask.sample(smp, in.uv);\n"
        "  if (m.a <= 0.0) return tex.sample(smp, in.uv);\n"
        "  return mix(tex.sample(smp, in.uv), plain.sample(smp, in.uv), m.a);\n"
        "}\n"
        // Selective AA pass (FXAA 3.11 console-style, masked): pixels inside
        // or adjacent to the exclusion mask (glyph/UI protection, ~1px
        // dilation) pass through untouched; everything else gets FXAA.
        "fragment float4 fsFxaaMasked(VOut in [[stage_in]],\n"
        "                             texture2d<float> tex [[texture(0)]],\n"
        "                             texture2d<float> maskTex [[texture(1)]],\n"
        "                             sampler smp [[sampler(0)]],\n"
        "                             constant float2 &texel [[buffer(2)]]) {\n"
        "  float2 uv = in.uv;\n"
        "  float mSelf = maskTex.sample(smp, uv).a;\n"
        "  float mR = maskTex.sample(smp, uv + float2( texel.x, 0.0)).a;\n"
        "  float mL = maskTex.sample(smp, uv + float2(-texel.x, 0.0)).a;\n"
        "  float mU = maskTex.sample(smp, uv + float2(0.0,  texel.y)).a;\n"
        "  float mD = maskTex.sample(smp, uv + float2(0.0, -texel.y)).a;\n"
        "  float mMax = max(max(mSelf, mR), max(mL, max(mU, mD)));\n"
        "  if (mMax > 0.05) return tex.sample(smp, uv);\n"
        "  float3 rgbNW = tex.sample(smp, uv + float2(-texel.x, -texel.y)).rgb;\n"
        "  float3 rgbNE = tex.sample(smp, uv + float2( texel.x, -texel.y)).rgb;\n"
        "  float3 rgbSW = tex.sample(smp, uv + float2(-texel.x,  texel.y)).rgb;\n"
        "  float3 rgbSE = tex.sample(smp, uv + float2( texel.x,  texel.y)).rgb;\n"
        "  float3 rgbM  = tex.sample(smp, uv).rgb;\n"
        "  float3 luma = float3(0.299, 0.587, 0.114);\n"
        "  float lumaNW = dot(rgbNW, luma), lumaNE = dot(rgbNE, luma);\n"
        "  float lumaSW = dot(rgbSW, luma), lumaSE = dot(rgbSE, luma);\n"
        "  float lumaM  = dot(rgbM, luma);\n"
        "  float lumaMin = min(lumaM, min(min(lumaNW, lumaNE), min(lumaSW, lumaSE)));\n"
        "  float lumaMax = max(lumaM, max(max(lumaNW, lumaNE), max(lumaSW, lumaSE)));\n"
        "  float2 dir = float2(-((lumaNW + lumaNE) - (lumaSW + lumaSE)),\n"
        "                       ((lumaNW + lumaSW) - (lumaNE + lumaSE)));\n"
        "  float dirReduce = max((lumaNW + lumaNE + lumaSW + lumaSE) * 0.25 * (1.0 / 12.0), 1.0 / 128.0);\n"
        "  float rcpDirMin = 1.0 / (min(abs(dir.x), abs(dir.y)) + dirReduce);\n"
        "  dir = clamp(dir * rcpDirMin, float2(-8.0, -8.0), float2(8.0, 8.0)) * texel;\n"
        "  float3 rgbA = 0.5 * (tex.sample(smp, uv + dir * (1.0/3.0 - 0.5)).rgb +\n"
        "                        tex.sample(smp, uv + dir * (1.0/3.0 + 0.5)).rgb);\n"
        "  float3 rgbB = rgbA * 0.5 + 0.25 * (tex.sample(smp, uv + dir * -0.5).rgb +\n"
        "                                      tex.sample(smp, uv + dir *  0.5).rgb);\n"
        "  float lumaB = dot(rgbB, luma);\n"
        "  if ((lumaB < lumaMin) || (lumaB > lumaMax)) return float4(rgbA, 1.0);\n"
        "  return float4(rgbB, 1.0);\n"
        "}\n";

    NSError *err = nil;
    id<MTLLibrary> lib = [p.device newLibraryWithSource:[NSString stringWithUTF8String:kMSL]
                                                options:nil
                                                  error:&err];
    if (!lib) {
        logLine("[MetalFX] disabled: shader library compilation failed: %s",
                err ? [[err localizedDescription] UTF8String] : "unknown");
        return false;
    }

    MTLRenderPipelineDescriptor *pd = [[MTLRenderPipelineDescriptor alloc] init];
    pd.vertexFunction = [lib newFunctionWithName:@"vs"];
    pd.fragmentFunction = [lib newFunctionWithName:@"fs"];
    pd.colorAttachments[0].pixelFormat = p.layer.pixelFormat;
    err = nil;
    p.pipeline = [p.device newRenderPipelineStateWithDescriptor:pd error:&err];
    if (!p.pipeline) {
        logLine("[MetalFX] disabled: pipeline creation failed: %s",
                err ? [[err localizedDescription] UTF8String] : "unknown");
        return false;
    }

    if (p.maskEnabled) {
        MTLRenderPipelineDescriptor *mpd = [[MTLRenderPipelineDescriptor alloc] init];
        mpd.vertexFunction = [lib newFunctionWithName:@"vs"];
        mpd.fragmentFunction = [lib newFunctionWithName:@"fsMasked"];
        mpd.colorAttachments[0].pixelFormat = p.layer.pixelFormat;
        err = nil;
        p.maskedPipeline = [p.device newRenderPipelineStateWithDescriptor:mpd error:&err];
        if (!p.maskedPipeline) {
            logLine("[MetalFX] disabled: masked pipeline creation failed: %s",
                    err ? [[err localizedDescription] UTF8String] : "unknown");
            return false;
        }
    }

    if (p.aaEnabled) {
        MTLRenderPipelineDescriptor *apd = [[MTLRenderPipelineDescriptor alloc] init];
        apd.vertexFunction = [lib newFunctionWithName:@"vs"];
        apd.fragmentFunction = [lib newFunctionWithName:@"fsFxaaMasked"];
        apd.colorAttachments[0].pixelFormat = kFramePixelFormat;
        err = nil;
        p.aaPipeline = [p.device newRenderPipelineStateWithDescriptor:apd error:&err];
        if (!p.aaPipeline) {
            logLine("[MetalFX] disabled: aa pipeline creation failed: %s",
                    err ? [[err localizedDescription] UTF8String] : "unknown");
            return false;
        }
    }

    MTLSamplerDescriptor *sd = [[MTLSamplerDescriptor alloc] init];
    sd.minFilter = MTLSamplerMinMagFilterLinear;
    sd.magFilter = MTLSamplerMinMagFilterLinear;
    sd.sAddressMode = MTLSamplerAddressModeClampToEdge;
    sd.tAddressMode = MTLSamplerAddressModeClampToEdge;
    p.sampler = [p.device newSamplerStateWithDescriptor:sd];
    if (!p.sampler) {
        logLine("[MetalFX] disabled: sampler creation failed");
        return false;
    }
    return true;
}

// Lanczos2 downscale with anti-ringing, port of upstream CuNNy
// magpie/Downscale.hlsl (CC0-1.0, commit 906031b). Used when the target
// content viewport is smaller than the 2x CNN output.
static const char *kDownscaleMSL =
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "kernel void cunny_downscale(texture2d<float, access::read> INPUT [[texture(0)]],\n"
    "                            texture2d<float, access::write> OUTPUT [[texture(1)]],\n"
    "                            uint2 gid [[thread_position_in_grid]]) {\n"
    "  const int2 osz = int2(OUTPUT.get_width(), OUTPUT.get_height());\n"
    "  if (gid.x >= uint(osz.x) || gid.y >= uint(osz.y)) return;\n"
    "  const int2 isz = int2(INPUT.get_width(), INPUT.get_height());\n"
    "  const float2 pt = float2(1.0f / float(isz.x), 1.0f / float(isz.y));\n"
    "  const float2 p = (float2(gid) + 0.5f) / float2(osz);\n"
    "  const float2 pp = p * float2(isz) - 0.5f;\n"
    "  const float2 p0 = metal::floor(pp);\n"
    "  const float2 f = pp - p0;\n"
    "  auto K = [&](float x) -> float {\n"
    "    const float kx = 3.1415926535897932f * x;\n"
    "    const float wx = 0.5f * kx;\n"
    "    return x < 1e-5f ? 1.0f : metal::sin(kx) * metal::sin(wx) / (x * x);\n"
    "  };\n"
    "  float4 wx = float4(K(1.0f + f.x), K(0.0f + f.x), K(1.0f - f.x), K(2.0f - f.x));\n"
    "  float4 wy = float4(K(1.0f + f.y), K(0.0f + f.y), K(1.0f - f.y), K(2.0f - f.y));\n"
    "  wx /= metal::dot(wx, float4(1.0f));\n"
    "  wy /= metal::dot(wy, float4(1.0f));\n"
    "  float3 vmin = float3(1e6f), vmax = float3(-1e6f);\n"
    "  float3 l[4][4];\n"
    "  for (int y = 0; y < 4; ++y) {\n"
    "    for (int x = 0; x < 4; ++x) {\n"
    "      int2 c = int2(metal::clamp(p0 + float2(x - 1, y - 1), float2(0.0f), float2(isz - 1)));\n"
    "      float3 q = INPUT.read(uint2(c)).rgb;\n"
    "      q = q * q; // D(x) = x*x\n"
    "      vmin = metal::min(vmin, q);\n"
    "      vmax = metal::max(vmax, q);\n"
    "      l[y][x] = q;\n"
    "    }\n"
    "  }\n"
    "  float3 v = float3(0.0f);\n"
    "  for (int y = 0; y < 4; ++y)\n"
    "    for (int x = 0; x < 4; ++x)\n"
    "      v += wy[y] * wx[x] * l[y][x];\n"
    "  v = metal::clamp(v, vmin, vmax);\n"
    "  OUTPUT.write(float4(metal::sqrt(v), 1.0f), gid);\n"
    "}\n";

// Initializes the CuNNy 2x CNN pipeline (upstream veryfast-NVL, LGPL-3.0).
// Failure only disables the CuNNy path; the presenter keeps working in
// MetalFX mode.
bool ensureCunny(Impl &p) {
    if (p.cunnyReady) {
        return true;
    }
    NSError *err = nil;
    id<MTLLibrary> lib = [p.device newLibraryWithSource:[NSString stringWithUTF8String:kCuNNyMSL]
                                                options:nil
                                                  error:&err];
    if (!lib) {
        logLine("[MetalFX] cunny disabled: shader compilation failed: %s",
                err ? [[err localizedDescription] UTF8String] : "unknown");
        return false;
    }
    id<MTLFunction> fnP1 = [lib newFunctionWithName:@"cunny_p1"];
    id<MTLFunction> fnP2 = [lib newFunctionWithName:@"cunny_p2"];
    id<MTLFunction> fnP3 = [lib newFunctionWithName:@"cunny_p3"];
    id<MTLFunction> fnP4 = [lib newFunctionWithName:@"cunny_p4"];
    if (!fnP1 || !fnP2 || !fnP3 || !fnP4) {
        logLine("[MetalFX] cunny disabled: missing compute functions");
        return false;
    }
    p.cunnyP1 = [p.device newComputePipelineStateWithFunction:fnP1 error:&err];
    p.cunnyP2 = [p.device newComputePipelineStateWithFunction:fnP2 error:&err];
    p.cunnyP3 = [p.device newComputePipelineStateWithFunction:fnP3 error:&err];
    p.cunnyP4 = [p.device newComputePipelineStateWithFunction:fnP4 error:&err];
    if (!p.cunnyP1 || !p.cunnyP2 || !p.cunnyP3 || !p.cunnyP4) {
        logLine("[MetalFX] cunny disabled: compute pipeline creation failed: %s",
                err ? [[err localizedDescription] UTF8String] : "unknown");
        return false;
    }

    id<MTLLibrary> dl = [p.device newLibraryWithSource:[NSString stringWithUTF8String:kDownscaleMSL]
                                                options:nil
                                                  error:&err];
    if (!dl) {
        logLine("[MetalFX] cunny disabled: downscale shader compilation failed: %s",
                err ? [[err localizedDescription] UTF8String] : "unknown");
        return false;
    }
    p.downscalePS = [p.device newComputePipelineStateWithFunction:[dl newFunctionWithName:@"cunny_downscale"]
                                                            error:&err];
    if (!p.downscalePS) {
        logLine("[MetalFX] cunny disabled: downscale pipeline creation failed: %s",
                err ? [[err localizedDescription] UTF8String] : "unknown");
        return false;
    }

    // Fixed-size resources: the CNN input is always the game framebuffer and
    // the output is always 2x, so these are created once and never rebuilt.
    const NSUInteger gw = (NSUInteger)p.gameW, gh = (NSUInteger)p.gameH;
    MTLTextureDescriptor *tDesc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                                                    width:gw
                                                                                   height:gh
                                                                                mipmapped:NO];
    tDesc.storageMode = MTLStorageModePrivate;
    tDesc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    p.cunnyT0 = [p.device newTextureWithDescriptor:tDesc];
    p.cunnyT1 = [p.device newTextureWithDescriptor:tDesc];
    p.cunnyT2 = [p.device newTextureWithDescriptor:tDesc];
    p.cunnyT3 = [p.device newTextureWithDescriptor:tDesc];

    MTLTextureDescriptor *oDesc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:kFramePixelFormat
                                                                                    width:gw * 2
                                                                                   height:gh * 2
                                                                                mipmapped:NO];
    oDesc.storageMode = MTLStorageModePrivate;
    oDesc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    if (p.cunnyScaler) {
        oDesc.usage |= p.cunnyScaler.colorTextureUsage;
    }
    p.cunnyOut = [p.device newTextureWithDescriptor:oDesc];

    if (!p.cunnyT0 || !p.cunnyT1 || !p.cunnyT2 || !p.cunnyT3 || !p.cunnyOut) {
        logLine("[MetalFX] cunny disabled: texture creation failed");
        return false;
    }

    p.cunnyReady = true;
    logLine("[MetalFX] cunny ready: model=veryfast-NVL source=funnyplanter/CuNNy@906031b input=%dx%d output=%dx%d",
            p.gameW, p.gameH, p.gameW * 2, p.gameH * 2);
    return true;
}

// Encodes the four CuNNy compute passes into `cb`.
bool encodeCunny(id<MTLCommandBuffer> cb, Impl &p, id<MTLTexture> input) {
    if (!cb || !input || !p.cunnyP1 || !p.cunnyP2 || !p.cunnyP3 || !p.cunnyP4 ||
        !p.cunnyT0 || !p.cunnyT1 || !p.cunnyT2 || !p.cunnyT3 || !p.cunnyOut) {
        return false;
    }
    id<MTLComputeCommandEncoder> ce = [cb computeCommandEncoder];
    if (!ce) {
        return false;
    }
    const MTLSize grid = MTLSizeMake((NSUInteger)p.gameW, (NSUInteger)p.gameH, 1);
    const MTLSize group = MTLSizeMake(8, 8, 1);

    [ce setComputePipelineState:p.cunnyP1];
    [ce setTexture:input atIndex:0];
    [ce setTexture:p.cunnyT0 atIndex:1];
    [ce setTexture:p.cunnyT1 atIndex:2];
    [ce dispatchThreads:grid threadsPerThreadgroup:group];

    [ce setComputePipelineState:p.cunnyP2];
    [ce setTexture:p.cunnyT0 atIndex:0];
    [ce setTexture:p.cunnyT1 atIndex:1];
    [ce setTexture:p.cunnyT2 atIndex:2];
    [ce setTexture:p.cunnyT3 atIndex:3];
    [ce dispatchThreads:grid threadsPerThreadgroup:group];

    [ce setComputePipelineState:p.cunnyP3];
    [ce setTexture:p.cunnyT2 atIndex:0];
    [ce setTexture:p.cunnyT3 atIndex:1];
    [ce setTexture:p.cunnyT0 atIndex:2];
    [ce dispatchThreads:grid threadsPerThreadgroup:group];

    [ce setComputePipelineState:p.cunnyP4];
    [ce setTexture:input atIndex:0];
    [ce setTexture:p.cunnyT0 atIndex:1];
    [ce setTexture:p.cunnyOut atIndex:2];
    [ce dispatchThreads:grid threadsPerThreadgroup:group];

    [ce endEncoding];
    return true;
}

// Encodes the lanczos2 downscale of `src` (2x) into `dst` (target size).
bool encodeDownscale(id<MTLCommandBuffer> cb, Impl &p, id<MTLTexture> src, id<MTLTexture> dst) {
    if (!cb || !p.downscalePS || !src || !dst) {
        return false;
    }
    id<MTLComputeCommandEncoder> ce = [cb computeCommandEncoder];
    if (!ce) {
        return false;
    }
    [ce setComputePipelineState:p.downscalePS];
    [ce setTexture:src atIndex:0];
    [ce setTexture:dst atIndex:1];
    const MTLSize grid = MTLSizeMake((NSUInteger)p.outW, (NSUInteger)p.outH, 1);
    const MTLSize group = MTLSizeMake(8, 8, 1);
    [ce dispatchThreads:grid threadsPerThreadgroup:group];
    [ce endEncoding];
    return true;
}

// Golden-validation helper: dumps the CNN input (CPU frame, BGRA) and the 2x
// GPU output (via blit readback) once, so the CPU reference implementation
// can be compared pixel-by-pixel.
void dumpCunnyOutput(Impl &p, SDL_Surface *frame, id<MTLCommandBuffer> cb) {
    char inPath[1024];
    snprintf(inPath, sizeof(inPath), "%s/input.bgra", p.cunnyDumpDir);
    FILE *f = fopen(inPath, "wb");
    if (f) {
        fwrite(frame->pixels, 1, (size_t)frame->pitch * frame->h, f);
        fclose(f);
    }
    const size_t bytes = (size_t)p.gameW * 2 * p.gameH * 2 * 4;
    id<MTLBuffer> buf = [p.device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    if (!buf) {
        logLine("[MetalFX] cunny dump failed: readback buffer creation failed");
        return;
    }
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
    if (!blit) {
        logLine("[MetalFX] cunny dump failed: blit encoder creation failed");
        return;
    }
    [blit copyFromTexture:p.cunnyOut
              sourceSlice:0
              sourceLevel:0
             sourceOrigin:MTLOriginMake(0, 0, 0)
               sourceSize:MTLSizeMake((NSUInteger)p.gameW * 2, (NSUInteger)p.gameH * 2, 1)
                 toBuffer:buf
         destinationOffset:0
    destinationBytesPerRow:(NSUInteger)p.gameW * 8
      destinationBytesPerImage:bytes];
    [blit endEncoding];
    NSString *dumpDir = [NSString stringWithUTF8String:p.cunnyDumpDir];
    [cb addCompletedHandler:^(id<MTLCommandBuffer> completed) {
        if (completed.status != MTLCommandBufferStatusCompleted) {
            return;
        }
        char outPath[1024];
        const char *dumpDirPath = [dumpDir UTF8String];
        snprintf(outPath, sizeof(outPath), "%s/output.bgra", dumpDirPath);
        FILE *fo = fopen(outPath, "wb");
        if (fo) {
            fwrite(buf.contents, 1, bytes, fo);
            fclose(fo);
            logLine("[MetalFX] cunny dumped: %s/input.bgra and %s", dumpDirPath, outPath);
        }
    }];
    p.cunnyDumped = true;
}

// Writes the alpha channel of a shared staging texture as an 8-bit BMP.
// Debug helper for verifying which pixels the text mask covers.
// Renders `src` through the masked selective-AA pipeline into `dst` at full
// texture size; `mask` gates which pixels may be processed.
bool encodeAAPass(id<MTLCommandBuffer> cb, Impl &p, id<MTLTexture> src, id<MTLTexture> dst,
                  id<MTLTexture> mask, id<MTLRenderPipelineState> pipeline,
                  float texelX, float texelY) {
    if (!cb || !src || !dst || !mask || !pipeline || !p.sampler) {
        return false;
    }
    MTLRenderPassDescriptor *rpd = [MTLRenderPassDescriptor renderPassDescriptor];
    rpd.colorAttachments[0].texture = dst;
    rpd.colorAttachments[0].loadAction = MTLLoadActionDontCare;
    rpd.colorAttachments[0].storeAction = MTLStoreActionStore;
    id<MTLRenderCommandEncoder> re = [cb renderCommandEncoderWithDescriptor:rpd];
    if (!re) {
        return false;
    }
    [re setRenderPipelineState:pipeline];
    [re setFragmentTexture:src atIndex:0];
    [re setFragmentTexture:mask atIndex:1];
    [re setFragmentSamplerState:p.sampler atIndex:0];
    const float texel[2] = {texelX, texelY};
    [re setFragmentBytes:texel length:sizeof(texel) atIndex:2];
    const float pos[8] = {-1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f};
    const float uvs[8] = {0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f};
    [re setVertexBytes:pos length:sizeof(pos) atIndex:0];
    [re setVertexBytes:uvs length:sizeof(uvs) atIndex:1];
    [re drawPrimitives:MTLPrimitiveTypeTriangleStrip vertexStart:0 vertexCount:4];
    [re endEncoding];
    return true;
}

bool dumpMaskToBMP(id<MTLTexture> staging, Impl &p) {
    if (staging.storageMode != MTLStorageModeShared) {
        return false;
    }
    const int w = p.gameW, h = p.gameH;
    std::vector<unsigned char> buf((size_t)h * w * 4);
    [staging getBytes:buf.data()
          bytesPerRow:(size_t)w * 4
             fromRegion:MTLRegionMake2D(0, 0, w, h)
            mipmapLevel:0];
    std::vector<unsigned char> bmp(54 + 1024 + (size_t)h * w);
    bmp[0] = 'B'; bmp[1] = 'M';
    uint32_t size = (uint32_t)bmp.size();
    memcpy(&bmp[2], &size, 4);
    uint32_t dataOff = 54 + 1024;
    memcpy(&bmp[10], &dataOff, 4);
    uint32_t dibSize = 40;
    memcpy(&bmp[14], &dibSize, 4);
    int32_t iw = w, ih = -h; // top-down
    memcpy(&bmp[18], &iw, 4);
    memcpy(&bmp[22], &ih, 4);
    uint16_t planes = 1, depth = 8;
    memcpy(&bmp[26], &planes, 2);
    memcpy(&bmp[28], &depth, 2);
    uint32_t biComp = 0;
    memcpy(&bmp[30], &biComp, 4);
    // Grayscale palette (256 entries).
    for (int i = 0; i < 256; ++i) {
        bmp[54 + i * 4] = (unsigned char)i;
        bmp[55 + i * 4] = (unsigned char)i;
        bmp[56 + i * 4] = (unsigned char)i;
    }
    const size_t out = 54 + 1024;
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            bmp[out + (size_t)y * w + x] = buf[((size_t)y * w + x) * 4 + 3];
        }
    }
    // Skip empty masks (e.g. title screens without a text layer) so the dump
    // only captures frames that actually contain glyphs.
    bool anyGlyph = false;
    for (size_t i = 0; i < (size_t)w * h; ++i) {
        if (bmp[out + i] != 0) { anyGlyph = true; break; }
    }
    if (!anyGlyph) {
        return false;
    }
    FILE *fp = fopen(p.maskDumpPath, "wb");
    if (!fp) {
        logLine("[MetalFX] mask dump failed: %s", p.maskDumpPath);
        return false;
    }
    fwrite(bmp.data(), 1, bmp.size(), fp);
    fclose(fp);
    logLine("[MetalFX] mask dumped: %s (%dx%d)", p.maskDumpPath, w, h);
    return true;
}

bool Presenter::present(SDL_Surface *frame) {
    Impl &p = *impl_;
    if (!p.active) {
        return false;
    }

    const int asyncFailure = p.asyncFailure->exchange(0);
    if (asyncFailure == 1 && p.cunnyReady) {
        logLine("[MetalFX] previous CuNNy GPU submission failed; falling back to MetalFX");
        p.cunnyReady = false;
        p.scalerMode = 0;
    } else if (asyncFailure != 0) {
        logLine("[MetalFX] previous GPU submission failed; disabling Metal presenter");
        p.active = false;
        return false;
    }

    p.frameCount++;
    CFAbsoluteTime now0 = CFAbsoluteTimeGetCurrent();
    if (now0 - p.lastFreqLog > 1.0) {
        logLine("[MetalFX] present fps: %.1f", (double)(p.frameCount - p.freqFrames) / (now0 - p.lastFreqLog));
        p.lastFreqLog = now0;
        p.freqFrames = p.frameCount;
    }
    int dw = 0, dh = 0;
    SDL_Metal_GetDrawableSize(p.window, &dw, &dh);
    if (dw <= 0 || dh <= 0) {
        logLine("[MetalFX] disabled: drawable size became invalid (%dx%d)", dw, dh);
        p.active = false;
        return false;
    }

    if (dw != p.drawW || dh != p.drawH) {
        p.drawW = dw;
        p.drawH = dh;
        if (!rebuildForOutputSize(*impl_)) {
            return false;
        }
    }

    if (!frame || frame->format->format != SDL_PIXELFORMAT_ARGB8888) {
        logLine("[MetalFX] disabled: unexpected framebuffer format 0x%x (expected ARGB8888)",
                frame ? (unsigned)frame->format->format : 0);
        p.active = false;
        return false;
    }

    if (!ensurePipeline(*impl_)) {
        p.active = false;
        return false;
    }

    if (!p.diagnosticsLogged) {
        const char *deviceName = p.device.name ? [p.device.name UTF8String] : "unknown";
        const char *scalerName = p.scalerMode == 1 ? "cunny" : (p.scalerMode == 2 ? "cunny+metalfx" : "metalfx");
        logLine("[MetalFX] init: device=%s game=%dx%d drawable=%dx%d out=%dx%d color=%s mode=%s scaler=%s textmask=%s aa=%s",
                deviceName, p.gameW, p.gameH, p.drawW, p.drawH, p.outW, p.outH,
                kPixelFormatBGRA8, p.colorMode == MTLFXSpatialScalerColorProcessingModePerceptual ? "perceptual" : "other",
                scalerName, p.maskEnabled ? "on" : "off",
                p.aaEnabled ? "masked-fxaa" : "off");
        p.diagnosticsLogged = true;
    }

    CFAbsoluteTime t0 = CFAbsoluteTimeGetCurrent();
    id<CAMetalDrawable> drawable = [p.layer nextDrawable];
    CFAbsoluteTime t1 = CFAbsoluteTimeGetCurrent();
    if (t1 - t0 > 0.02) {
        logLine("[MetalFX] nextDrawable slow: %.3fs (frame %d)", t1 - t0, p.frameCount);
    }
    if (!drawable) {
        logLine("[MetalFX] disabled: drawable acquisition failed");
        p.active = false;
        return false;
    }

    id<MTLCommandBuffer> cb = [p.queue commandBuffer];
    if (!cb) {
        logLine("[MetalFX] disabled: command buffer creation failed");
        p.active = false;
        return false;
    }

    // Upload the whole CPU frame through a shared staging texture; the blit
    // keeps GPU ordering safe without waiting on the CPU. A per-frame staging
    // allocation is acceptable at ONS's event-driven refresh rates.
    MTLTextureDescriptor *stDesc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:kFramePixelFormat
                                                                                      width:(NSUInteger)p.gameW
                                                                                     height:(NSUInteger)p.gameH
                                                                                  mipmapped:NO];
    stDesc.storageMode = MTLStorageModeShared;
    stDesc.usage = MTLTextureUsageShaderRead;
    id<MTLTexture> staging = [p.device newTextureWithDescriptor:stDesc];
    if (!staging) {
        logLine("[MetalFX] disabled: staging texture creation failed");
        p.active = false;
        return false;
    }
    [staging replaceRegion:MTLRegionMake2D(0, 0, p.gameW, p.gameH)
               mipmapLevel:0
                 withBytes:frame->pixels
               bytesPerRow:frame->pitch];

    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
    if (!blit) {
        logLine("[MetalFX] disabled: blit encoder creation failed");
        p.active = false;
        return false;
    }
    [blit copyFromTexture:staging
              sourceSlice:0
              sourceLevel:0
             sourceOrigin:MTLOriginMake(0, 0, 0)
               sourceSize:MTLSizeMake(p.gameW, p.gameH, 1)
                toTexture:p.inputTexture
         destinationSlice:0
         destinationLevel:0
        destinationOrigin:MTLOriginMake(0, 0, 0)];

    // Upload the text layer alpha mask (full-screen, same layout as the
    // frame) so the final pass can keep post-processing off glyph shapes.
    bool maskReady = false;
    bool maskHasContent = false;
    if (p.maskEnabled && p.maskTexture) {
        MTLTextureDescriptor *maskStDesc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:kFramePixelFormat
                                                                                             width:(NSUInteger)p.gameW
                                                                                            height:(NSUInteger)p.gameH
                                                                                         mipmapped:NO];
        maskStDesc.storageMode = MTLStorageModeShared;
        maskStDesc.usage = MTLTextureUsageShaderRead;
        id<MTLTexture> maskStaging = [p.device newTextureWithDescriptor:maskStDesc];
        if (maskStaging) {
            if (p.testMask) {
                // Synthetic diagonal stripes to exercise the masked composite
                // without depending on a game's text layer.
                const int bpp = 4;
                const size_t rowBytes = (size_t)p.gameW * bpp;
                std::vector<unsigned char> pattern((size_t)p.gameH * rowBytes, 0);
                for (int y = 0; y < p.gameH; ++y) {
                    for (int x = 0; x < p.gameW; ++x) {
                        if (((x / 32) + (y / 32)) % 2 == 0) {
                            size_t off = (size_t)y * rowBytes + (size_t)x * bpp;
                            pattern[off + 3] = 255; // alpha = mask on
                        }
                    }
                }
                [maskStaging replaceRegion:MTLRegionMake2D(0, 0, p.gameW, p.gameH)
                               mipmapLevel:0
                                 withBytes:pattern.data()
                               bytesPerRow:rowBytes];
                maskReady = true;
                maskHasContent = true;
            } else if (p.textLayer) {
                SDL_Surface *tl = p.textLayer;
                if (tl->w == p.gameW && tl->h == p.gameH &&
                    tl->format->format == SDL_PIXELFORMAT_ARGB8888) {
                    // Quick CPU scan for glyph pixels; an empty text layer
                    // means no dynamic text this frame, so selective AA must
                    // stay off rather than degrade into full-frame FXAA.
                    const unsigned char *px = (const unsigned char *)tl->pixels;
                    for (int y = 0; y < p.gameH && !maskHasContent; ++y) {
                        const unsigned char *row = px + (size_t)y * tl->pitch;
                        for (int x = 0; x < p.gameW; ++x) {
                            if (row[x * 4 + 3] != 0) { maskHasContent = true; break; }
                        }
                    }
                    [maskStaging replaceRegion:MTLRegionMake2D(0, 0, p.gameW, p.gameH)
                                   mipmapLevel:0
                                     withBytes:tl->pixels
                                   bytesPerRow:tl->pitch];
                    maskReady = true;
                } else if (!p.diagnosticsLogged) {
                    logLine("[MetalFX] mask skipped: text layer %dx%d fmt=0x%x", tl->w, tl->h,
                            (unsigned)tl->format->format);
                }
            }
            if (maskReady) {
                [blit copyFromTexture:maskStaging
                          sourceSlice:0
                          sourceLevel:0
                         sourceOrigin:MTLOriginMake(0, 0, 0)
                           sourceSize:MTLSizeMake(p.gameW, p.gameH, 1)
                            toTexture:p.maskTexture
                     destinationSlice:0
                     destinationLevel:0
                    destinationOrigin:MTLOriginMake(0, 0, 0)];
                if (p.maskDumpPath && !p.maskDumped) {
                    p.maskDumped = dumpMaskToBMP(maskStaging, p);
                }
            }
        }
    }
    [blit endEncoding];

    // Selective AA runs at game resolution on the unmasked pixels only. Feed
    // the result into either scaler, then protect glyphs again in the final
    // composite so neither CuNNy nor MetalFX reconstructs the text layer.
    id<MTLTexture> scalerInput = p.inputTexture;
    bool aaActive = p.aaEnabled && maskHasContent && p.useScaler;
    if (aaActive) {
        if (!encodeAAPass(cb, p, p.inputTexture, p.aaTexture, p.maskTexture, p.aaPipeline,
                          1.0f / (float)p.gameW, 1.0f / (float)p.gameH)) {
            p.active = false;
            return false;
        }
        scalerInput = p.aaTexture;
    } else if (p.aaEnabled && !p.aaDisabledLogged && !maskHasContent) {
        logLine("[MetalFX] selective AA off: empty exclusion mask (no text layer this frame)");
        p.aaDisabledLogged = true;
    }

    id<MTLTexture> srcTexture = scalerInput;

    // CuNNy 2x CNN path (scalerMode 1 or 2), only when actually upscaling.
    bool cunnyActive = p.cunnyReady && p.scalerMode != 0 && p.useScaler;
    bool encodedCunny = false;
    if (cunnyActive) {
        if (!encodeCunny(cb, p, scalerInput)) {
            logLine("[MetalFX] cunny disabled: encode failed; falling back to MetalFX");
            p.cunnyReady = false;
            cunnyActive = false;
        } else {
            encodedCunny = true;
            const int c2w = p.gameW * 2, c2h = p.gameH * 2;
            if (abs(p.outW - c2w) <= 2 && abs(p.outH - c2h) <= 2) {
                // target ~= 2x: CuNNy -> final pass directly.
                srcTexture = p.cunnyOut;
            } else if (p.outW > c2w || p.outH > c2h) {
                if (p.scalerMode == 2) {
                    // target > 2x: CuNNy -> MetalFX -> final.
                    if (!p.cunnyScaler) {
                        logLine("[MetalFX] cunny+metalfx scaler missing; using CuNNy only");
                        p.scalerMode = 1;
                        srcTexture = p.cunnyOut;
                    } else {
                        p.cunnyScaler.colorTexture = p.cunnyOut;
                        p.cunnyScaler.inputContentWidth = (NSUInteger)c2w;
                        p.cunnyScaler.inputContentHeight = (NSUInteger)c2h;
                        p.cunnyScaler.outputTexture = p.outputTexture;
                        [p.cunnyScaler encodeToCommandBuffer:cb];
                        srcTexture = p.outputTexture;
                    }
                } else {
                    // CuNNy only: final pass linearly upscales 2x -> target.
                    srcTexture = p.cunnyOut;
                }
            } else {
                // target < 2x: CuNNy -> high-quality downscale -> final.
                if (!encodeDownscale(cb, p, p.cunnyOut, p.downscaleOut)) {
                    logLine("[MetalFX] cunny disabled: downscale encode failed; falling back to MetalFX");
                    p.cunnyReady = false;
                    cunnyActive = false;
                } else {
                    srcTexture = p.downscaleOut;
                }
            }
            if (cunnyActive && p.cunnyDumpDir && !p.cunnyDumped) {
                dumpCunnyOutput(p, frame, cb);
            }
        }
    }

    if (!cunnyActive) {
        if (p.useScaler) {
            p.scaler.colorTexture = srcTexture;
            p.scaler.inputContentWidth = (NSUInteger)p.gameW;
            p.scaler.inputContentHeight = (NSUInteger)p.gameH;
            p.scaler.outputTexture = p.outputTexture;
            [p.scaler encodeToCommandBuffer:cb];
            srcTexture = p.outputTexture;
        }
    }

    MTLRenderPassDescriptor *rpd = [MTLRenderPassDescriptor renderPassDescriptor];
    rpd.colorAttachments[0].texture = drawable.texture;
    rpd.colorAttachments[0].loadAction = MTLLoadActionClear;
    rpd.colorAttachments[0].clearColor = MTLClearColorMake(0.0, 0.0, 0.0, 1.0);
    rpd.colorAttachments[0].storeAction = MTLStoreActionStore;
    id<MTLRenderCommandEncoder> re = [cb renderCommandEncoderWithDescriptor:rpd];
    if (!re) {
        logLine("[MetalFX] disabled: render encoder creation failed");
        p.active = false;
        return false;
    }
    if (maskReady && p.useScaler) {
        // Text-protected composite: processed image everywhere except glyph
        // areas, which sample the plain input with linear scaling.
        [re setRenderPipelineState:p.maskedPipeline];
        [re setFragmentTexture:srcTexture atIndex:0];
        [re setFragmentTexture:p.inputTexture atIndex:1];
        [re setFragmentTexture:p.maskTexture atIndex:2];
    } else {
        [re setRenderPipelineState:p.pipeline];
        [re setFragmentTexture:srcTexture atIndex:0];
    }
    [re setFragmentSamplerState:p.sampler atIndex:0];

    // Letterboxed quad in NDC (Metal: y up, origin top-left in pixel space).
    const float left = -1.0f + 2.0f * (float)p.outX / (float)p.drawW;
    const float right = -1.0f + 2.0f * (float)(p.outX + p.outW) / (float)p.drawW;
    const float top = 1.0f - 2.0f * (float)p.outY / (float)p.drawH;
    const float bottom = 1.0f - 2.0f * (float)(p.outY + p.outH) / (float)p.drawH;
    const float pos[8] = {left, top, right, top, left, bottom, right, bottom};
    const float uvs[8] = {0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f};
    [re setVertexBytes:pos length:sizeof(pos) atIndex:0];
    [re setVertexBytes:uvs length:sizeof(uvs) atIndex:1];
    [re drawPrimitives:MTLPrimitiveTypeTriangleStrip vertexStart:0 vertexCount:4];
    [re endEncoding];

    p.layer.displaySyncEnabled = YES;
    [cb presentDrawable:drawable];
    const bool submittedWithCunny = encodedCunny;
    std::shared_ptr<std::atomic<int>> failureSignal = p.asyncFailure;
    [cb addCompletedHandler:^(id<MTLCommandBuffer> completed) {
        if (completed.status != MTLCommandBufferStatusCompleted) {
            failureSignal->store(submittedWithCunny ? 1 : 2);
            const char *message = completed.error ? [[completed.error localizedDescription] UTF8String] : "unknown";
            logLine("[MetalFX] command buffer failed: %s", message);
        }
    }];
    if (p.frameCount % 60 == 0) {
        // Periodic GPU timing for A/B (macOS 11+ API).
        const int submittedScalerMode = p.scalerMode;
        [cb addCompletedHandler:^(id<MTLCommandBuffer> completed) {
            if (completed.status == MTLCommandBufferStatusCompleted && completed.GPUEndTime > 0) {
                logLine("[MetalFX] gpu frame time: %.3f ms (scaler=%d)",
                        (completed.GPUEndTime - completed.GPUStartTime) * 1000.0, submittedScalerMode);
            }
        }];
    }
    [cb commit];
    CFAbsoluteTime now1 = CFAbsoluteTimeGetCurrent();
    if (now1 - now0 > 0.02) {
        logLine("[MetalFX] present frame slow: %.3fs (drawable %.3fs)", now1 - now0, t1 - t0);
    }
    return true;
}

} // namespace onsyuri_metalfx
