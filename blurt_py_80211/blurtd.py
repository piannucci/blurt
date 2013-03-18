#!/usr/bin/env python
import blurt
import socket
import os, os.path
import time
import pylab as pl

pl.ion()

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
pl.figure(frameon=False, figsize=(5,2))

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
                pl.axes((0,0,1,1), frameon=False, axisbg=(0,0,0,0))
                pl.tick_params(bottom=False, top=False, left=False, right=False)
                pl.specgram(input, Fs=Fs, cmap='custom_cmap', NFFT=64,
                            noverlap=64-4, window=lambda x:x,
                            interpolation='nearest')
                pl.xlim(0,input.size/(Fs))
                pl.ylim(-Fs/2, Fs/2)
                pl.draw()
                import StringIO, base64
                s = StringIO.StringIO()
                pl.savefig(s, format='png', dpi=72, transparent=True)
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
