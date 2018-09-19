//
//  Visualization.m
//  BlurtDemo
//
//  Created by Peter Iannucci on 11/14/15.
//  Copyright © 2015 MIT. All rights reserved.
//

#import "Visualization.h"
#include <OpenGL/gl.h>
#import <GLKit/GLKit.h>
#import <GLUT/GLUT.h>
#import <Accelerate/Accelerate.h>

static Visualization *openGLView;

#define STFTSize 2048
#define STFTSizeLog2 11
#define STFTStride 1024

const int spectrogramWidth = 1536, spectrogramHeight = STFTSize/2;
const double spectrogramRate = 240; // STFTs per second
const float gridSpacingTime = .5;

struct List {
    struct List *next, *prev;
};

struct STFTBuffer {
    struct List listHead;
    Float32 powerSpectrum[STFTSize/2];
    double timestampSeconds;
};

typedef struct STFTBuffer STFTBuffer;

@interface Visualization () {
    CVDisplayLinkRef displayLink;
    char *screenPixels;
    GLushort colors[4][256];
    SoundRecorder *recorder;
    struct List stftListHead;
    STFTBuffer *lastSTFT;
    Float32 bufferedSamples[STFTSize-1];
    int bufferedSampleCount;
    FFTSetup fftSetup;
    GLuint tickLabelTextures[20];
    float tickLabelX[20];
    float tickLabelT[20];
    GLuint screenTex;
    CTFontRef font;
    NSDictionary *textAttributes;
    char *stringBitmap;
    int stringBitmapSize, stringTexWidth, stringTexHeight;
    int stringContentWidth, stringContentHeight;
    GLsizei pixelWidth, pixelHeight;
    GLsizei pointWidth, pointHeight;
    CGSize devicePixel, devicePoint;
    BOOL sizeChanged;
    NSObject *glSync, *stftSync;
}
@end

@implementation Visualization

- (void)drawString:(NSString *)string
{
    NSAttributedString *as = [[NSAttributedString alloc] initWithString:string attributes:textAttributes];
    CTLineRef line = CTLineCreateWithAttributedString((CFAttributedStringRef)as);
    CGFloat fAscent, descent, leading;
    double fWidth = CTLineGetTypographicBounds(line, &fAscent, &descent, &leading);
    float scale = [self convertSizeToBacking:NSMakeSize(1., 1.)].width;
    
    int ascent = fAscent + 1;
    int width = ceilf(fWidth*scale), height = ceilf((ascent + descent)*scale);
    stringContentWidth = width;
    stringContentHeight = height;
    width = 1<<(int)ceilf(log2f(width));
    height = 1<<(int)ceilf(log2f(height));
    
    if (width*height > stringBitmapSize)
    {
        stringBitmapSize = width*height;
        if (stringBitmap)
            free(stringBitmap);
        stringBitmap = (char *)malloc(stringBitmapSize);
    }
    
    CGContextRef stringBitmapContext = CGBitmapContextCreate(stringBitmap, width, height, 8, width, nil, (CGBitmapInfo)kCGImageAlphaOnly);
    memset(stringBitmap, 0, width*height);
    CGContextScaleCTM(stringBitmapContext, scale, scale);
    CGContextSetTextPosition(stringBitmapContext, 0., height - scale*ascent);
    CTLineDraw(line, stringBitmapContext);
    CFRelease(line);
    CGContextRelease(stringBitmapContext);
    
    glTexImage2D(GL_TEXTURE_2D, 0, GL_ALPHA, width, height, 0, GL_ALPHA, GL_UNSIGNED_BYTE, stringBitmap);
    stringTexWidth = width;
    stringTexHeight = height;
}


- (void) awakeFromNib
{
    glSync = [NSObject new];
    stftSync = [NSObject new];
    font = CTFontCreateUIFontForLanguage(kCTFontUIFontEmphasizedSystem, 0.0, NULL);
    textAttributes = @{(__bridge id)kCTFontAttributeName: (__bridge id)font};
    stftListHead.next = stftListHead.prev = &stftListHead;
    bufferedSampleCount = 0;
    openGLView = self;
    screenPixels = malloc(spectrogramWidth*spectrogramHeight);
    for (int i=0; i<256; i++)
    {
        colors[0][i] = MAX(MIN(1.5 - 4.5*fabs(i/256.-.77), 1.), 0.) * 0xffff;
        colors[1][i] = MAX(MIN(1.5 - 4.5*fabs(i/256.-.55), 1.), 0.) * 0xffff;
        colors[2][i] = MAX(MIN(1.5 - 4.5*fabs(i/256.-.33), 1.), 0.) * 0xffff;
        colors[3][i] = 0xffff;
    }
    for (int i=0; i<spectrogramWidth*spectrogramHeight; i++)
        screenPixels[i] = (i/spectrogramWidth) ^ (i%spectrogramWidth);
    
    fftSetup = vDSP_create_fftsetup(STFTSizeLog2, 0);
    
    recorder = [SoundRecorder new];
    recorder.delegate = self;
    recorder.inBufSize = 64;
    [recorder startRecordingAtRate:96e3];
}

float blackman(int n, int N)
{
    float theta = (2*M_PI*n)/(N-1);
    const float alpha = .16;
    const float a0 = 1-alpha/2, a1 = .5, a2 = alpha/2;
    return a0 - a1 * cosf(theta) + a2 * cosf(2*theta);
}

- (BOOL)soundRecorder:(SoundRecorder *)soundRecorder recordedFrames:(void *)frames withCount:(size_t)frameCount basicDescription:(AudioStreamBasicDescription)asbd
{
    Float32 *samples = (Float32 *)frames;
    frameCount*=2; // it boggles my mind that this is necessary
    int bufferedCount = bufferedSampleCount;
    int workingBufferSize = bufferedCount + (int)frameCount;
    Float32 workingBuffer[workingBufferSize];
    memcpy(workingBuffer, bufferedSamples, sizeof(Float32)*bufferedCount);
    for (int i=0; i<frameCount; i++)
        workingBuffer[bufferedCount + i] = samples[i]; // this is exactly not what the docs say to do
    
    Float32 window[STFTSize];
    Float32 *real = window, *imag = window+1;
    DSPSplitComplex split = {real, imag};
    
    int cursor = 0;
    while (cursor + STFTSize <= workingBufferSize)
    {
        memcpy(window, workingBuffer, sizeof(Float32) * STFTSize);
        cursor += STFTStride;
        
        // multiply by window function
        for (int i=0; i<STFTSize; i++)
            window[i] *= blackman(i, STFTSize);
        
        vDSP_fft_zrip(fftSetup, &split, 2, STFTSizeLog2, kFFTDirection_Forward);
        
        STFTBuffer *buffer = malloc(sizeof(STFTBuffer));
        for (int i=0; i<STFTSize/2; i++)
            buffer->powerSpectrum[i] = 20*log10f(cabsf(CMPLXF(real[i], imag[i])));
        buffer->timestampSeconds = 0;
        
        @synchronized(stftSync) {
            buffer->listHead.next = &stftListHead;
            buffer->listHead.prev = stftListHead.prev;
            stftListHead.prev->next = &buffer->listHead;
            stftListHead.prev = &buffer->listHead;
        }
    }
    if (cursor < workingBufferSize)
    {
        bufferedSampleCount = bufferedCount + (int)frameCount - cursor;
        memcpy(bufferedSamples, workingBuffer+cursor, sizeof(Float32)*bufferedSampleCount);
    }
    else
        bufferedSampleCount = 0;
    
    return NO;
}

static CVReturn MyDisplayLinkCallback(CVDisplayLinkRef displayLink, const CVTimeStamp* now, const CVTimeStamp* outputTime, CVOptionFlags flagsIn, CVOptionFlags* flagsOut, void* displayLinkContext)
{
    CVReturn result = [(__bridge Visualization *)displayLinkContext getFrameForTime:outputTime];
    return result;
}

- (void)advanceToTime:(double)time
{
    static int lastFrame = -spectrogramWidth;
    int frame = time * spectrogramRate;
    int shift = frame - lastFrame;
    if (shift > spectrogramWidth || shift < 0)
    {
        shift = spectrogramWidth;
        lastFrame = frame-spectrogramWidth;
    }
    for (int y=0; y<spectrogramHeight; y++)
        memmove(screenPixels + spectrogramWidth*y, screenPixels + spectrogramWidth*y + shift, spectrogramWidth - shift);
    struct List *buffer;
    for (int x=lastFrame; x<frame; x++)
    {
        @synchronized(stftSync) {
            buffer = stftListHead.next;
            stftListHead.next = buffer->next;
            stftListHead.next->prev = &stftListHead;
        }
        if (buffer != &stftListHead)
        {
            if (lastSTFT)
                free(lastSTFT);
            lastSTFT = (STFTBuffer *)buffer;
        }
        if (lastSTFT)
        {
            for (int y=0; y<spectrogramHeight; y++)
                screenPixels[spectrogramWidth*(y+1) + x - frame] = MIN(MAX(lastSTFT->powerSpectrum[y]*3.0 + 16, 0) * 1.5, 0xff);
        }
        else
        {
            for (int y=0; y<spectrogramHeight; y++)
                screenPixels[spectrogramWidth*(y+1) + x - frame] = 0;
        }
    }
    lastFrame = frame;
}

- (CVReturn)getFrameForTime:(const CVTimeStamp*)outputTime
{
    [self.openGLContext makeCurrentContext];
    
    @synchronized(glSync) {
        if (sizeChanged)
        {
            [self.openGLContext makeCurrentContext];
            glViewport(0, 0, pixelWidth, pixelHeight);
            glLoadIdentity();
            glOrtho(0, pointWidth, 0, pointHeight, -1, 1);
            sizeChanged = NO;
        }
    }
    
    double time = (double)outputTime->videoTime / outputTime->videoTimeScale;
    static int startTime = 0;
    if (startTime == 0)
        startTime = time;
    time -= startTime;
    
    float columnsTraveled = time*spectrogramRate;
    [self advanceToTime:time];
    
    glPixelMapusv(GL_PIXEL_MAP_I_TO_R, 256, colors[0]);
    glPixelMapusv(GL_PIXEL_MAP_I_TO_G, 256, colors[1]);
    glPixelMapusv(GL_PIXEL_MAP_I_TO_B, 256, colors[2]);
    glPixelMapusv(GL_PIXEL_MAP_I_TO_A, 256, colors[3]);
    
    glActiveTexture(GL_TEXTURE0);
    
    glClearColor(0, 0, 0, 0);
    glClear(GL_COLOR_BUFFER_BIT);

    NSRect spectrogram = NSMakeRect(1, 30, pointWidth-1, pointHeight-30);
    
    glPushMatrix();
    glTranslatef(spectrogram.origin.x, spectrogram.origin.y, 0.);
    glScalef(spectrogram.size.width, spectrogram.size.height, 1.);
    // draw spectrogram
    {
        glEnable(GL_TEXTURE_2D);
        glBindTexture(GL_TEXTURE_2D, screenTex);
        glPixelTransferi(GL_MAP_COLOR, GL_TRUE);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, spectrogramWidth, spectrogramHeight, GL_COLOR_INDEX, GL_UNSIGNED_BYTE, screenPixels);
        glDrawArrays(GL_TRIANGLES, 0, 6);
        glDisable(GL_TEXTURE_2D);
    }
    // draw border
    {
        glLineWidth(2);
        glColor3f(0xff, 0xff, 0xff);
        const GLint indices[4] = {0,3,5,4};
        glDrawElements(GL_LINE_LOOP, 4, GL_UNSIGNED_INT, indices);
    }
    // draw horizontal grid lines
    {
        const GLint indices[4] = {0,4,3,5};
        glPushMatrix();
        glLineWidth(1);
        int scrollAmount = ((float)pixelWidth/spectrogramWidth) * columnsTraveled;
        glLineStipple(1, ((0xff00ff00 << (scrollAmount % 16)) >> 16) & 0xffff);
        glEnable(GL_LINE_STIPPLE);
        
        glTranslatef(0., -.25, 0.);
        glDrawElements(GL_LINES, 2, GL_UNSIGNED_INT, indices);
        
        glTranslatef(0., -.25, 0.);
        glDrawElements(GL_LINES, 2, GL_UNSIGNED_INT, indices);
        
        glTranslatef(0., -.25, 0.);
        glDrawElements(GL_LINES, 2, GL_UNSIGNED_INT, indices);

        glDisable(GL_LINE_STIPPLE);
        
        glPopMatrix();
    }
    // draw vertical grid lines
    {
        glEnable(GL_LINE_STIPPLE);
        const GLint indices[4] = {0,3};
        glPushMatrix();
        float spectrogramTime = (float)spectrogramWidth / spectrogramRate;
        float rightMarginTime = time;
        float leftMarginTime = rightMarginTime - spectrogramTime;
        float firstTickTime = floorf(leftMarginTime/gridSpacingTime)*gridSpacingTime;
        float firstTickX = (firstTickTime - leftMarginTime) / spectrogramTime - 1;
        float gridSpacingX = gridSpacingTime / spectrogramTime;
        
        glTranslatef(firstTickX, 0., 0.);
        glLineStipple(1, 0xff00);

        float x = firstTickX, t = firstTickTime;
        while (t <= rightMarginTime+gridSpacingTime)
        {
            glDrawElements(GL_LINES, 2, GL_UNSIGNED_INT, indices);
            glTranslatef(gridSpacingX, 0., 0.);
            t += gridSpacingTime;
            x += gridSpacingX;
        }
        glPopMatrix();
        glDisable(GL_LINE_STIPPLE);

        x = firstTickX; t = firstTickTime;
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glEnable(GL_BLEND);
        glEnable(GL_TEXTURE_2D);
        float scale = [self convertSizeToBacking:NSMakeSize(1., 1.)].width;
        while (t <= rightMarginTime+gridSpacingTime)
        {
            glBindTexture(GL_TEXTURE_2D,  tickLabelTextures[0]);
            glPixelTransferi(GL_MAP_COLOR, GL_FALSE);
            [self drawString:[NSString stringWithFormat:@"%.1f s", t]];

            glColor3f(1., 1., 1.);
            glPushMatrix();
            glTranslatef(x+1 - .5*stringContentWidth/scale/spectrogram.size.width, -16/spectrogram.size.height, 0.);
            glScalef(stringTexWidth/scale/spectrogram.size.width, -stringTexHeight/scale/spectrogram.size.height, 1.);
            glDrawArrays(GL_TRIANGLES, 0, 6);
            
            glPopMatrix();
            
            t += gridSpacingTime;
            x += gridSpacingX;
        }
        
        glDisable(GL_TEXTURE_2D);
        glDisable(GL_BLEND);
    }
    glPopMatrix();

    [[self openGLContext] flushBuffer];
    
    return kCVReturnSuccess;
}

-(void)setFrame:(NSRect)frame
{
    [super setFrame:frame];
    
    if (displayLink)
        CVDisplayLinkStop(displayLink);
    
    @synchronized(glSync) {
        NSRect bounds = [self bounds];
        NSRect backingBounds = [self convertRectToBacking:bounds];
        pixelWidth = (GLsizei)backingBounds.size.width,
        pixelHeight = (GLsizei)backingBounds.size.height;
        pointWidth = (GLsizei)bounds.size.width;
        pointHeight = (GLsizei)bounds.size.height;
        
        devicePixel = CGSizeMake(2./pixelWidth, 2./pixelHeight);
        float scale = [self convertSizeToBacking:NSMakeSize(1., 1.)].width;
        devicePoint = CGSizeMake(devicePixel.width/scale, devicePixel.height/scale);
        
        sizeChanged = YES;
    }
    
    if (displayLink)
        CVDisplayLinkStart(displayLink);
    
    NSLog(@"%d x %d pixels", pointWidth, pointHeight);
}

-(void)prepareOpenGL
{
    [self setFrame:self.frame];
    
    glActiveTexture(GL_TEXTURE0);
    
    glGenTextures(20, tickLabelTextures);
    for (int i=0; i<20; i++)
    {
        glBindTexture(GL_TEXTURE_2D, tickLabelTextures[i]);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    }
    static const GLshort vertexData[] = {+1, +1, 1, 1, +0, +1, 0, 1, +1, +0, 1, 0, +1, +0, 1, 0, +0, +1, 0, 1, +0, +0, 0, 0};
    glEnableClientState(GL_VERTEX_ARRAY);
    glVertexPointer(2, GL_SHORT, 8, vertexData);
    glEnableClientState(GL_TEXTURE_COORD_ARRAY);
    glClientActiveTexture(GL_TEXTURE0);
    glTexCoordPointer(2, GL_SHORT, 8, vertexData+2);
    
    glGenTextures(1, &screenTex);
    glBindTexture(GL_TEXTURE_2D, screenTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, spectrogramWidth, spectrogramHeight, 0, GL_COLOR_INDEX, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    
    GLint swapInt = 1;
    [self.openGLContext setValues:&swapInt forParameter:NSOpenGLCPSwapInterval];
    
    CVDisplayLinkCreateWithActiveCGDisplays(&displayLink);
    CVDisplayLinkSetOutputCallback(displayLink, &MyDisplayLinkCallback, (__bridge void *)(self));
    
    CGLContextObj cglContext = [[self openGLContext] CGLContextObj];
    CGLPixelFormatObj cglPixelFormat = [[self pixelFormat] CGLPixelFormatObj];
    CVDisplayLinkSetCurrentCGDisplayFromOpenGLContext(displayLink, cglContext, cglPixelFormat);
    NSLog(@"Actual output video refresh period: %f", CVDisplayLinkGetActualOutputVideoRefreshPeriod(displayLink));
    CVTime time = CVDisplayLinkGetNominalOutputVideoRefreshPeriod(displayLink);
    NSLog(@"Nominal output video refresh period: %f", (double)time.timeValue / time.timeScale);
    
    CVDisplayLinkStart(displayLink);
}

- (void)dealloc
{
    CVDisplayLinkRelease(displayLink);
}

@end
