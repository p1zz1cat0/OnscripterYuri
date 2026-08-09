// MetalFX Spatial MVP presenter for OnscripterYuri (macOS only).
//
// Takes the already-composited CPU frame (accumulation_surface, ARGB8888),
// uploads it to host-owned Metal textures, runs the selected MetalFX/CuNNy
// scaling pipeline, and presents the result through a final render pass onto
// the SDL-created CAMetalLayer drawable.
//
// The SDL renderer path stays untouched: when this presenter is inactive or
// fails, ONScripter continues to use SDL_UpdateTexture/RenderCopy/RenderPresent.

#ifndef ONSYURI_METALFX_PRESENTER_H
#define ONSYURI_METALFX_PRESENTER_H

struct SDL_Window;
struct SDL_Surface;

namespace onsyuri_metalfx {

struct Impl;

class Presenter {
public:
    // Takes ownership of nothing; the metal view stays owned by SDL.
    Presenter(SDL_Window *window, void *metal_view, int game_width, int game_height);
    ~Presenter();

    // True when initialization succeeded and the Metal presenter is active.
    bool isActive() const;

    // Uploads the full CPU frame and presents it. Returns false when the
    // session is (or just became) inactive; the caller must fall back to the
    // SDL path for the current frame.
    bool present(SDL_Surface *frame);

    // Supplies the current full-screen text layer (text_info.image_surface,
    // same pixel format and size as the game frame). Its alpha channel masks
    // glyph shapes; the final pass shows the plain linear-upscaled frame in
    // masked regions so post-processing never touches the text.
    void setTextLayer(SDL_Surface *textLayer);

private:
    Impl *impl_;
};

} // namespace onsyuri_metalfx

#endif // ONSYURI_METALFX_PRESENTER_H
