#!/usr/bin/env python
try:
    import _coreaudio
    del _coreaudio
except:
    from distutils.core import setup, Extension
    from Cython.Distutils import build_ext
    import numpy, os
    wd = os.getcwd()
    os.chdir(os.path.dirname(__file__))
    setup(
        name="_coreaudio", version="0.1",
        ext_modules=[
             Extension("_coreaudio", ["_coreaudio.pyx"], extra_link_args=["-framework", "CoreAudio"])
             ],
        include_dirs = [numpy.get_include()],
        cmdclass = {'build_ext': build_ext},
        script_args = ['build']
    )
    builddir = 'build'
    libdir = os.path.join(builddir, [x for x in os.listdir(builddir) if x.startswith('lib')][0])
    resultname = [x for x in os.listdir(libdir) if x.startswith('_coreaudio')][0]
    resultpath = os.path.join(libdir, resultname)
    os.rename(resultpath, resultname)
    os.chdir(wd)
    del numpy, build_ext, setup, Extension, os

from coreaudio import play, record, play_and_record, AudioInterface
import stream
