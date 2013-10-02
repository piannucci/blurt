//
//  BLDetailViewController.h
//  Blurt Mobile
//
//  Created by Peter Iannucci on 6/24/13.
//  Copyright (c) 2013 Peter Iannucci. All rights reserved.
//

#import <UIKit/UIKit.h>
#import <AVFoundation/AVFoundation.h>

@interface BLDetailViewController : UITableViewController

@property (strong, nonatomic) id detailItem;

@property (weak, nonatomic) IBOutlet UILabel *nameLabel, *publicIDLabel, *otpLabel, *privateIDLabel, *secretKeyLabel;
@property (nonatomic, strong) AVAudioPlayer *audioPlayer;

@end
