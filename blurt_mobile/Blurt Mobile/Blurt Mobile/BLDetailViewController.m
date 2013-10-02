//
//  BLDetailViewController.m
//  Blurt Mobile
//
//  Created by Peter Iannucci on 6/24/13.
//  Copyright (c) 2013 Peter Iannucci. All rights reserved.
//

#import <CommonCrypto/CommonCrypto.h>
#import "BLAppDelegate.h"
#import "dsp.h"
#import "BLDetailViewController.h"

const static char *hex_table = "0123456789abcdef";
const static char *modhex_table = "cbdefghijklnrtuv";

NSString *bin2modhex(const uint8_t *bytes, size_t length) {
    NSMutableString *string = [NSMutableString stringWithCapacity:2*length];
    for (int i=0; i<length; i++)
        [string appendFormat:@"%c%c", modhex_table[(bytes[i] >> 4) & 15], modhex_table[bytes[i] & 15]];
    return string;
}

NSString *bin2hex(const uint8_t *bytes, size_t length) {
    NSMutableString *string = [NSMutableString stringWithCapacity:3*length];
    for (int i=0; i<length; i++)
        [string appendFormat:@"%c%c ", hex_table[(bytes[i] >> 4) & 15], hex_table[bytes[i] & 15]];
    return string;
}

uint16_t crc(uint8_t *bytes, size_t length) {
    uint16_t crc = 0xffff;
    for (int i=0; i<length; i++) {
        crc ^= bytes[i];
        for (int j=0; j<8; j++)
            crc = (crc >> 1) ^ ((crc & 1) ? 0x8408 : 0);
    }
    return crc;
}

void fix_crc(uint8_t *bytes, size_t length) {
    uint16_t residue = crc(bytes, length-2) ^ 0xffff;
    bytes[length-2] = residue & 0xff;
    bytes[length-1] = (residue>>8) & 0xff;
}

NSData *generateWavFile(NSData *waveform) {
    NSMutableData *wavFile = [NSMutableData dataWithLength:waveform.length+44];
    void *wavFileBytes = wavFile.mutableBytes;
    OSWriteBigInt32   (wavFileBytes, 0, 'RIFF');
    OSWriteLittleInt32(wavFileBytes, 4, waveform.length + 36);
    OSWriteBigInt32   (wavFileBytes, 8, 'WAVE');
    OSWriteBigInt32   (wavFileBytes, 12, 'fmt ');
    OSWriteLittleInt32(wavFileBytes, 16, 16);
    OSWriteLittleInt16(wavFileBytes, 20, 1);
    OSWriteLittleInt16(wavFileBytes, 22, 1);
    OSWriteLittleInt32(wavFileBytes, 24, 48000);
    OSWriteLittleInt32(wavFileBytes, 28, 48000*2);
    OSWriteLittleInt16(wavFileBytes, 32, 2);
    OSWriteLittleInt16(wavFileBytes, 34, 16);
    OSWriteBigInt32   (wavFileBytes, 36, 'data');
    OSWriteLittleInt32(wavFileBytes, 40, waveform.length);
    memcpy(wavFileBytes+44, waveform.bytes, waveform.length);
    return wavFile;
}

NSData *encodeOneTimePassword(NSData *publicID,
                              NSData *privateID,
                              NSData *secretKey,
                              uint64_t *sessionCounter,
                              uint32_t *useCounter,
                              uint32_t *timestamp) {
    uint8_t plaintext[16], ciphertext[16];
    memcpy(plaintext+0, privateID.bytes, 6);
    OSWriteLittleInt16(plaintext, 6, sessionCounter);
    OSWriteLittleInt32(plaintext, 8, timestamp);
    plaintext[11] = useCounter;
    arc4random_buf(plaintext+12, 2);
    fix_crc(plaintext, 16);
    
    CCCrypt(kCCEncrypt, kCCAlgorithmAES128, kCCOptionECBMode, secretKey.bytes, secretKey.length, NULL, plaintext, 16, ciphertext, 16, NULL);
    
    uint8_t otp[22];
    memcpy(otp, publicID.bytes, 6);
    memcpy(otp+6, ciphertext, 16);
    return [NSData dataWithBytes:otp length:22];
}

@interface BLDetailViewController ()

@end

@implementation BLDetailViewController

- (id)initWithNibName:(NSString *)nibNameOrNil bundle:(NSBundle *)nibBundleOrNil
{
    self = [super initWithNibName:nibNameOrNil bundle:nibBundleOrNil];
    if (self) {
        // Custom initialization
    }
    return self;
}

- (void)viewDidLoad
{
    [super viewDidLoad];
	// Do any additional setup after loading the view.
}

- (void)viewWillAppear:(BOOL)animated {
    self.navigationItem.title = [self.detailItem valueForKey:@"shortName"];
    self.nameLabel.text = [self.detailItem valueForKey:@"longName"];
    
    uint8_t bytes[16];
    
    [[self.detailItem valueForKey:@"publicID"] getBytes:bytes length:6];
    self.publicIDLabel.text = bin2modhex(bytes, 6);
    
    [[self.detailItem valueForKey:@"privateID"] getBytes:bytes length:6];
    self.privateIDLabel.text = bin2hex(bytes, 6);
    
    [[self.detailItem valueForKey:@"secretKey"] getBytes:bytes length:16];
    self.secretKeyLabel.text = bin2hex(bytes, 16);
    
    self.otpLabel.text = @"Tap to generate...";
    self.otpLabel.textColor = [UIColor grayColor];
}

- (void)didReceiveMemoryWarning
{
    [super didReceiveMemoryWarning];
    // Dispose of any resources that can be recreated.
}

- (void)tableView:(UITableView *)tableView didSelectRowAtIndexPath:(NSIndexPath *)indexPath {
    if (indexPath.section == 1 && indexPath.row == 0) {
        NSData *publicID = [self.detailItem valueForKey:@"publicID"];
        NSData *privateID = [self.detailItem valueForKey:@"privateID"];
        NSData *secretKey = [self.detailItem valueForKey:@"secretKey"];
        uint64_t sessionCounter = [[self.detailItem valueForKey:@"sessionCounter"] unsignedLongLongValue];
        uint32_t useCounter = [[self.detailItem valueForKey:@"useCounter"] unsignedLongLongValue];
        
        BLAppDelegate *appDelegate = [UIApplication sharedApplication].delegate;
        // if this is a new session, increment sessionCounter and randomize timestamp_base
        if (![appDelegate.credentialsInThisSession containsObject:self.detailItem]) {
            [appDelegate.credentialsInThisSession addObject:self.detailItem];
            [self.detailItem setValue:[NSNumber numberWithUnsignedLongLong:++sessionCounter] forKey:@"sessionCounter"];
            [self.detailItem setValue:[NSDate dateWithTimeIntervalSinceNow:arc4random_uniform(1<<24)/8.f]forKey:@"timestampBase"];
        }
        
        // increment useCounter
        [self.detailItem setValue:[NSNumber numberWithUnsignedLongLong:++useCounter] forKey:@"useCounter"];
        
        [appDelegate saveContext];
        
        // compute timestamp
        uint32_t timestamp = (int32_t)((-[[self.detailItem valueForKey:@"timestampBase"] timeIntervalSinceNow]) * 8.0) & 0xffffff;
        
        NSData *otp = encodeOneTimePassword(publicID, privateID, secretKey, sessionCounter, useCounter, timestamp);
        NSData *waveform = encodeBlurtWaveform(otp, 0);
        NSData *wavFile = generateWavFile(waveform);
        self.audioPlayer = [[AVAudioPlayer alloc] initWithData:wavFile error:NULL];
        [[AVAudioSession sharedInstance] setCategory:AVAudioSessionCategoryPlayAndRecord
                                         withOptions:AVAudioSessionCategoryOptionDefaultToSpeaker | AVAudioSessionCategoryOptionMixWithOthers
                                               error:NULL];
        self.audioPlayer.volume = 1.0;
        [self.audioPlayer prepareToPlay];
        [self.audioPlayer play];
        
        self.otpLabel.text = bin2modhex(otp.bytes, otp.length);
        self.otpLabel.textColor = [UIColor blackColor];
        [tableView deselectRowAtIndexPath:indexPath animated:YES];
    }
}

@end
