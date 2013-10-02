//
//  BLCredentialsListController.h
//  Blurt Mobile
//
//  Created by Peter Iannucci on 6/24/13.
//  Copyright (c) 2013 Peter Iannucci. All rights reserved.
//

#import <UIKit/UIKit.h>
#import <CoreData/CoreData.h>
#import "BLDetailViewController.h"

@interface BLCredentialsListController : UITableViewController <NSFetchedResultsControllerDelegate>

@property (strong, nonatomic) BLDetailViewController *detailViewController;

@property (strong, nonatomic) NSFetchedResultsController *fetchedResultsController;
@property (strong, nonatomic) NSManagedObjectContext *managedObjectContext;

@end
