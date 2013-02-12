from distutils.core import setup, Extension
from Cython.Distutils import build_ext
import numpy
setup(
    name="_coreaudio", version="0.1",
    ext_modules=[
         Extension("_coreaudio", ["_coreaudio.pyx"], extra_link_args=["-framework", "CoreAudio"])
         ],
    include_dirs = [numpy.get_include()],
    cmdclass = {'build_ext': build_ext}
)


