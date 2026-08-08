#import <AppKit/AppKit.h>

static NSString *YoghourtEnvironmentValue(NSString *key) {
    NSString *value = NSProcessInfo.processInfo.environment[key];
    return value.length > 0 ? value : nil;
}

static NSImage *YoghourtRoundedDockIcon(NSImage *source) {
    NSSize size = source.size;
    NSImage *result = [[NSImage alloc] initWithSize:size];
    [result lockFocus];
    CGFloat radius = MIN(size.width, size.height) * 0.08;
    [[NSBezierPath bezierPathWithRoundedRect:NSMakeRect(0, 0, size.width, size.height)
                                    xRadius:radius
                                    yRadius:radius] addClip];
    [source drawInRect:NSMakeRect(0, 0, size.width, size.height)
              fromRect:NSZeroRect
             operation:NSCompositingOperationCopy
              fraction:1.0];
    [result unlockFocus];
    return result;
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

extern "C" void YoghourtApplyPresentation(void) {
    @autoreleasepool {
        NSString *title = YoghourtEnvironmentValue(@"YOGHOURT_GAME_TITLE");
        if (title) {
            NSProcessInfo.processInfo.processName = title;
            NSMenuItem *applicationMenu = NSApp.mainMenu.itemArray.firstObject;
            if (applicationMenu) applicationMenu.title = title;
        }

        NSString *iconPath = YoghourtEnvironmentValue(@"YOGHOURT_GAME_ICON");
        if (iconPath) {
            NSImage *icon = [[NSImage alloc] initWithContentsOfFile:iconPath];
            if (icon) NSApp.applicationIconImage = YoghourtRoundedDockIcon(icon);
#if !__has_feature(objc_arc)
            [icon release];
#endif
        }
    }
}
