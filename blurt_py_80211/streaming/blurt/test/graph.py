import queue
import blurt
import numpy as np
import pylab as pl
c = blurt.phy.ieee80211a.Channel(96e3, 17e3, 8)
e = blurt.phy.ieee80211a.IEEE80211aEncoderBlock(c)
d = blurt.phy.ieee80211a.IEEE80211aDecoderBlock(c)
e.input_queues = (queue.Queue(),)
e.output_queues = (queue.Queue(),)
d.input_queues = (queue.Queue(),)
d.output_queues = (queue.Queue(),)
d.nChannelsPerFrame = 1
e.intermediate_upsample = 1
e.start()
d.start()
if 1:
    length = 4
    np.random.seed(1)
    for i in range(100):
        datagram = bytes(np.random.randint(0, 255, length, dtype=np.uint8))
        e.input_queues[0].put(datagram)
        e.process()
        waveform = e.output_queues[0].get()[:,0,None]
        d.input_queues[0].put((waveform, 0, 0))
        d.process()
        try:
            print(d.output_queues[0].get_nowait()[0] == datagram)
        except queue.Empty:
            print(False)
elif 0:
    e.input_queues[0].put(b'\0\0\0\0')
    e.input_queues[0].put(b'\0\0\0\0')
    e.process()
    d.input_queues[0].put((e.output_queues[0].get()[:,0,None], 0, 0))
    d.input_queues[0].put((e.output_queues[0].get()[:,0,None], 0, 0))
    d.process()
    print(d.output_queues[0].get_nowait()[0])
    print(d.output_queues[0].empty())
elif 0:
    e.input_queues[0].put(b'\0\0\0\0')
    e.input_queues[0].put(b'\0\0\0\0')
    e.process()
    y = np.r_[e.output_queues[0].get(), e.output_queues[0].get()]
    pl.clf()
    _=pl.specgram(y[:,0], Fs=96e3, vmin=-48-30, NFFT=512, noverlap=512-16)
elif 0:
    e.input_queues[0].put(b'\0\0\0\0')
    e.input_queues[0].put(b'\0\0\0\0')
    e.process()
    y = np.r_[e.output_queues[0].get(), e.output_queues[0].get()]
    d.input_queues[0].put((y[:,0,None], 0, 0))
    d.process()
    pl.clf()
    _=pl.specgram(y[:,0], Fs=96e3, vmin=-48-30, NFFT=512, noverlap=512-16)
    print(d.output_queues[0].get_nowait()[0])
    print(d.output_queues[0].empty())
