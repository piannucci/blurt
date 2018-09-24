import sys

_clearLine = b'\r\x1b[2K'

class ScrollingScreen:
    def __init__(self, f):
        self.f = f
        self.statusLine = b''
        self.buffer = []

    def write(self, text):
        self.buffer.append(text)
        if text.endswith('\n'):
            self.flush()

    def flush(self):
        text = ''.join(self.buffer).encode()
        self.buffer.clear()
        self.f.write(_clearLine + text + self.statusLine)

    def status(self, text):
        self.statusLine = text.encode()
        self.f.write(_clearLine + self.statusLine)

    def __dealloc__(self):
        self.f.write(_clearLine)

sys.stderr = ScrollingScreen(sys.stderr.buffer.raw)
