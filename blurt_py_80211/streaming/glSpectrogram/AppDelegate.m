//
//  AppDelegate.m
//  BlurtDemo
//
//  Created by Peter Iannucci on 11/14/15.
//  Copyright © 2015 MIT. All rights reserved.
//

#import "AppDelegate.h"

@interface AppDelegate ()

@property (weak) IBOutlet NSWindow *window;
@property (weak) IBOutlet NSView *glView;
@end

@implementation AppDelegate

- (void)applicationDidFinishLaunching:(NSNotification *)aNotification {
    [self.window center];
    [self.window makeFirstResponder:self.glView];
    [self.window makeKeyAndOrderFront:nil];
    self.window.backgroundColor = [NSColor blackColor];
    
    if (!(_window.styleMask & NSFullScreenWindowMask))
        [_window toggleFullScreen:nil];
    self.window.titlebarAppearsTransparent = YES;
}

- (BOOL)applicationShouldTerminateAfterLastWindowClosed:(NSApplication *)theApplication {
    return YES;
}

@end
