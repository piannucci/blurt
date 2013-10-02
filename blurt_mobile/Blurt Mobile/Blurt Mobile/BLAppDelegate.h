//
//  BLAppDelegate.h
//  Blurt Mobile
//
//  Created by Peter Iannucci on 6/22/13.
//  Copyright (c) 2013 Peter Iannucci. All rights reserved.
//

#import <UIKit/UIKit.h>

@interface BLAppDelegate : UIResponder <UIApplicationDelegate>

@property (strong, nonatomic) UIWindow *window;

@property (readonly, strong, nonatomic) NSManagedObjectContext *managedObjectContext;
@property (readonly, strong, nonatomic) NSManagedObjectModel *managedObjectModel;
@property (readonly, strong, nonatomic) NSPersistentStoreCoordinator *persistentStoreCoordinator;
@property (strong, nonatomic) NSMutableSet *credentialsInThisSession;

- (void)saveContext;
- (NSURL *)applicationDocumentsDirectory;

@end
