#!/usr/bin/env python
import socket
from http.server import HTTPServer, SimpleHTTPRequestHandler

class HTTPServerV6(HTTPServer):
    address_family = socket.AF_INET6

HTTPServerV6(('::', 8080), SimpleHTTPRequestHandler).serve_forever()
