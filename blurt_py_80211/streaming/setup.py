#!/usr/bin/env python

from setuptools import setup, find_packages, Extension
import numpy

setup(
    name='Blurt',
    version='0.2',
    description='Data over audio',
    author='Peter Iannucci',
    author_email='iannucci@mit.edu',
    url='https://github.com/piannucci/blurt',
    packages=find_packages(),
    ext_modules=[
        Extension(
            'blurt.phy.kernels',
            ['src/kernels.cpp'],
            include_dirs=['/opt/local/include', numpy.get_include()],
            extra_compile_args=['-std=c++14']
        )
    ],
)
