#import <AppKit/AppKit.h>
#import <SDL2/SDL_syswm.h>

#import "YoghourtDockIcon.h"
#import "YoghourtWindowPresentation.h"

extern "C" void YoghourtONSWindowLayoutChanged(int width, int height, bool fullscreen);

static NSString *YoghourtEnvironmentValue(NSString *key) {
    NSString *value = NSProcessInfo.processInfo.environment[key];
    return value.length > 0 ? value : nil;
}

extern "C" void YoghourtPreparePresentation(void) {
    @autoreleasepool {
        NSString *title = YoghourtEnvironmentValue(@"YOGHOURT_GAME_TITLE");
        if (title) {
            setprogname(title.UTF8String);
            NSProcessInfo.processInfo.processName = title;
        }
    }
}

extern "C" void YoghourtApplyPresentation(SDL_Window *sdlWindow) {
    @autoreleasepool {
        NSString *title = YoghourtEnvironmentValue(@"YOGHOURT_GAME_TITLE");
        if (title) {
            NSProcessInfo.processInfo.processName = title;
            NSMenuItem *applicationMenu = NSApp.mainMenu.itemArray.firstObject;
            if (applicationMenu) applicationMenu.title = title;
        }

        NSString *iconPath = YoghourtEnvironmentValue(@"YOGHOURT_GAME_ICON");
        if (iconPath) {
            NSImage *icon = YoghourtLoadDockIcon(iconPath);
            if (icon) NSApp.applicationIconImage = icon;
        }

        SDL_SysWMinfo info;
        SDL_VERSION(&info.version);
        if (sdlWindow && SDL_GetWindowWMInfo(sdlWindow, &info) == SDL_TRUE) {
            YoghourtConfigureGameWindow((__bridge void *)info.info.cocoa.window,
                                        YoghourtONSWindowLayoutChanged);
        }
    }
}
