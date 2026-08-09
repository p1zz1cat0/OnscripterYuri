// Thin SDL adapter for Yoghourt's engine-independent SpatialPresenter.

#include "MetalFXPresenter.h"

#include <YoghourtSpatialPresenter.h>

#include <SDL.h>
#include <SDL_metal.h>
#include <QuartzCore/CAMetalLayer.h>

#include <cstring>

namespace onsyuri_metalfx {

struct Impl {
    yoghourt_spatial::Presenter *presenter = nullptr;
    SDL_Surface *textLayer = nullptr;
};

namespace {

yoghourt_spatial::ScalerMode scalerModeFromEnvironment() {
    const char *value = getenv("YOGHOURT_ONS_METALFX_SCALER");
    if (value && strcmp(value, "cunny") == 0) {
        return yoghourt_spatial::ScalerMode::cuNNy;
    }
    if (value && strcmp(value, "cunny+metalfx") == 0) {
        return yoghourt_spatial::ScalerMode::cuNNyPlusMetalFX;
    }
    return yoghourt_spatial::ScalerMode::metalFX;
}

bool isBGRA8Frame(const SDL_Surface *surface, int width, int height) {
    return surface && surface->pixels && surface->w == width &&
        surface->h == height &&
        surface->format->format == SDL_PIXELFORMAT_ARGB8888 &&
        surface->pitch >= width * 4;
}

yoghourt_spatial::CPUFrame frameView(const SDL_Surface *surface) {
    return {
        surface->pixels,
        surface->w,
        surface->h,
        static_cast<size_t>(surface->pitch),
        yoghourt_spatial::PixelFormat::bgra8Unorm,
    };
}

} // namespace

Presenter::Presenter(SDL_Window *window, void *metal_view, int game_width, int game_height)
    : impl_(new Impl) {
    if (!window || !metal_view || game_width <= 0 || game_height <= 0) {
        return;
    }
    CAMetalLayer *layer = (__bridge CAMetalLayer *)SDL_Metal_GetLayer(
        static_cast<SDL_MetalView>(metal_view));
    if (!layer) {
        fputs("[SpatialPresenter] ONS adapter disabled: SDL_Metal_GetLayer failed\n", stdout);
        fflush(stdout);
        return;
    }

    yoghourt_spatial::Options options;
    options.scaler = scalerModeFromEnvironment();
    const char *mask = getenv("YOGHOURT_ONS_METALFX_TEXTMASK");
    options.enableOverlayMask = !mask || strcmp(mask, "0") != 0;
    const char *maskDump = getenv("YOGHOURT_ONS_METALFX_MASKDUMP");
    options.maskDumpPath = maskDump && maskDump[0] ? maskDump : nullptr;
    const char *testMask = getenv("YOGHOURT_ONS_METALFX_TESTMASK");
    options.testMask = testMask && strcmp(testMask, "1") == 0;
    const char *cuNNyDump = getenv("YOGHOURT_ONS_METALFX_CUNNY_DUMP");
    options.cuNNyDumpDirectory = cuNNyDump && cuNNyDump[0] ? cuNNyDump : nullptr;

    impl_->presenter = new yoghourt_spatial::Presenter(
        (__bridge void *)layer, game_width, game_height, options);
}

Presenter::~Presenter() {
    delete impl_->presenter;
    delete impl_;
}

bool Presenter::isActive() const {
    return impl_->presenter && impl_->presenter->isActive();
}

void Presenter::setTextLayer(SDL_Surface *textLayer) {
    impl_->textLayer = textLayer;
}

bool Presenter::present(SDL_Surface *frame) {
    if (!impl_->presenter || !frame ||
        !isBGRA8Frame(frame, frame->w, frame->h)) {
        return false;
    }
    const yoghourt_spatial::CPUFrame source = frameView(frame);
    if (impl_->textLayer &&
        isBGRA8Frame(impl_->textLayer, frame->w, frame->h)) {
        const yoghourt_spatial::CPUFrame overlay = frameView(impl_->textLayer);
        return impl_->presenter->present(source, &overlay);
    }
    return impl_->presenter->present(source);
}

} // namespace onsyuri_metalfx
