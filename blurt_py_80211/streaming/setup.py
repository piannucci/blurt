#!/usr/bin/env python

from distutils.core import setup, Extension
import numpy

setup(
    name='Blurt',
    version='0.2',
    description='Data over audio',
    author='Peter Iannucci',
    author_email='iannucci@mit.edu',
    url='https://github.com/piannucci/blurt',
    packages=['blurt', 'blurt.audio', 'blurt.phy', 'blurt.mac', 'blurt.net', 'blurt.tools'],
    ext_modules=[
        Extension(
            'blurt.phy.kernels',
            ['src/kernels.cpp'],
            include_dirs=['/opt/local/include', numpy.get_include()],
            extra_compile_args=['-std=c++14']
        )
    ],
)
