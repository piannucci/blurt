#!/usr/bin/env python
import matplotlib, matplotlib.mlab
matplotlib.use('Agg')

import blurt
import socket
import os, os.path
import time
import pylab as pl
import numpy as np

np.seterr(divide='ignore')

#pl.ion()

sockfile = "/blurt/socket"

if os.path.exists(sockfile):
    os.remove(sockfile)

server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
server.bind(sockfile)
os.chmod(sockfile, 0775)
server.listen(5)

cmap = {
    'red':   ((0.0, 1.0, 1.0), (1.0, 1.0, 1.0)),
    'green': ((0.0, 1.0, 1.0), (1.0, 1.0, 1.0)),
    'blue':  ((0.0, 1.0, 1.0), (1.0, 1.0, 1.0)),
    'alpha': ((0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (1.0, 1.0, 1.0))
}

pl.register_cmap(name='custom_cmap', data=cmap)
pl.figure(frameon=False, figsize=(6,3))

try:
    while True:
        conn, addr = server.accept()
        data = conn.recv(1024)
        if data.startswith('/Library/WebServer/Documents/uploads') and data.endswith('.wav'):
            try:
                input, Fs = blurt.util.readwave(data)
                upsample_factor = blurt.upsample_factor * Fs / float(blurt.Fs)
                input = blurt.audioLoopback.processInput(input, Fs, blurt.Fc, upsample_factor)
                Fs /= upsample_factor
                pl.clf()
                pl.axis('off')
                ax = pl.axes((0,0,1,1), frameon=False, axisbg=(0,0,0,0))
                pl.tick_params(bottom=False, top=False, left=False, right=False)
                Pxx,freqs,bins = matplotlib.mlab.specgram(input, Fs=Fs, NFFT=64, noverlap=64-4, window=lambda x:x)
                Pxx = 10.*np.log10(Pxx)[::-1,:,np.newaxis]
                Pxx -= Pxx.min()
                Pxx /= Pxx.max()
                Pxx = np.clip(Pxx * 2. - 1., 0., 1.)
                Pxx = np.concatenate((np.ones_like(Pxx)/Pxx,)*3 + (Pxx,), axis=2)
                ax.imshow(Pxx, interpolation='nearest', vmin=0, vmax=1,
                          extent=(0, bins.max(), freqs[0], freqs[-1]))
                ax.axis('auto')
                pl.xlim(0,bins.max())
                pl.ylim(freqs[0], freqs[-1])
                pl.draw()
                import StringIO, base64
                s = StringIO.StringIO()
                pl.savefig(s, format='png', dpi=64, transparent=True)
                img = 'data:image/png;base64,' + base64.b64encode(s.getvalue())
                s.close()
                results, _ = blurt.wifi.decode(input)
                output = []
                for result in results:
                    payload, _, _, lsnr_estimate = result
                    output.append(repr(''.join(map(chr, payload))) + (' @ %.3f dB' % lsnr_estimate))
                output = '\n'.join(output) or 'No packets decoded'
                response = '%08d%s%08d%s' % (len(img), img, len(output), output)
                conn.send('%08d%s' % (len(response), response))
                #conn.send(output)
            except Exception, e:
                print repr(e)
                conn.send('Decoder error')
        conn.close()
        print data
except KeyboardInterrupt:
    pass

server.close()
os.remove(sockfile)
