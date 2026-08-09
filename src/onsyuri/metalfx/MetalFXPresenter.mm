// Objective-C++ implementation of the MetalFX Spatial presenter.
// Compiled only on macOS (config_darwin in CMakeLists.txt).

#include "MetalFXPresenter.h"

#include <Metal/Metal.h>
#include <MetalFX/MetalFX.h>
#include <QuartzCore/CAMetalLayer.h>
#include <SDL.h>
#include <SDL_metal.h>

#include <algorithm>
#include <cmath>
#include <cstring>
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

    impl_->active = true;
}

Presenter::~Presenter() {
    if (impl_->maskDumpPath) {
        free(impl_->maskDumpPath);
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
        aaDesc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
        p.aaTexture = [p.device newTextureWithDescriptor:aaDesc];
        if (!p.aaTexture) {
            logLine("[MetalFX] disabled: aa texture creation failed");
            p.active = false;
            return false;
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

// Writes the alpha channel of a shared staging texture as an 8-bit BMP.
// Debug helper for verifying which pixels the text mask covers.
// Renders `src` through the masked selective-AA pipeline into `dst` at full
// texture size; `mask` gates which pixels may be processed.
bool encodeAAPass(id<MTLCommandBuffer> cb, Impl &p, id<MTLTexture> src, id<MTLTexture> dst,
                  id<MTLTexture> mask, id<MTLRenderPipelineState> pipeline,
                  float texelX, float texelY) {
    MTLRenderPassDescriptor *rpd = [MTLRenderPassDescriptor renderPassDescriptor];
    rpd.colorAttachments[0].texture = dst;
    rpd.colorAttachments[0].loadAction = MTLLoadActionDontCare;
    rpd.colorAttachments[0].storeAction = MTLStoreActionStore;
    id<MTLRenderCommandEncoder> re = [cb renderCommandEncoderWithDescriptor:rpd];
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
        logLine("[MetalFX] init: device=%s game=%dx%d drawable=%dx%d out=%dx%d color=%s mode=%s scaler=%s textmask=%s aa=%s",
                deviceName, p.gameW, p.gameH, p.drawW, p.drawH, p.outW, p.outH,
                kPixelFormatBGRA8, p.colorMode == MTLFXSpatialScalerColorProcessingModePerceptual ? "perceptual" : "other",
                p.useScaler ? "yes" : "no", p.maskEnabled ? "on" : "off",
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

    // Selective AA runs at game resolution on the unmasked pixels only; the
    // masked (protected) pixels pass through untouched. The result feeds
    // MetalFX, which still processes the whole frame including text.
    id<MTLTexture> srcTexture = p.inputTexture;
    bool aaActive = p.aaEnabled && maskHasContent && p.useScaler;
    if (aaActive) {
        if (!encodeAAPass(cb, p, p.inputTexture, p.aaTexture, p.maskTexture, p.aaPipeline,
                          1.0f / (float)p.gameW, 1.0f / (float)p.gameH)) {
            p.active = false;
            return false;
        }
        srcTexture = p.aaTexture;
    } else if (p.aaEnabled && !p.aaDisabledLogged && !maskHasContent) {
        logLine("[MetalFX] selective AA off: empty exclusion mask (no text layer this frame)");
        p.aaDisabledLogged = true;
    }

    if (p.useScaler) {
        p.scaler.colorTexture = srcTexture;
        p.scaler.inputContentWidth = (NSUInteger)p.gameW;
        p.scaler.inputContentHeight = (NSUInteger)p.gameH;
        p.scaler.outputTexture = p.outputTexture;
        [p.scaler encodeToCommandBuffer:cb];
        srcTexture = p.outputTexture;
    }

    MTLRenderPassDescriptor *rpd = [MTLRenderPassDescriptor renderPassDescriptor];
    rpd.colorAttachments[0].texture = drawable.texture;
    rpd.colorAttachments[0].loadAction = MTLLoadActionClear;
    rpd.colorAttachments[0].clearColor = MTLClearColorMake(0.0, 0.0, 0.0, 1.0);
    rpd.colorAttachments[0].storeAction = MTLStoreActionStore;
    id<MTLRenderCommandEncoder> re = [cb renderCommandEncoderWithDescriptor:rpd];
    if (!aaActive && maskReady && p.useScaler) {
        // Legacy text-protected composite: MetalFX everywhere except glyph
        // areas, which sample the plain input (linear upscale).
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
    [cb commit];
    CFAbsoluteTime now1 = CFAbsoluteTimeGetCurrent();
    if (now1 - now0 > 0.02) {
        logLine("[MetalFX] present frame slow: %.3fs (drawable %.3fs)", now1 - now0, t1 - t0);
    }
    return true;
}

} // namespace onsyuri_metalfx
