import sys
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

# 1. Custom Build Command to handle Lazy Numpy Import
class CustomBuildExt(build_ext):
    def finalize_options(self):
        build_ext.finalize_options(self)
        # Prevent numpy from being imported before it's installed
        import builtins
        builtins.__NUMPY_SETUP__ = False
        import numpy
        # Add numpy include dir to the extension
        self.include_dirs.append(numpy.get_include())

# 2. Define Extension (Sources only, no include_dirs yet)
ext_modules = [
    Extension(
        "_quant_svcj",
        sources=["_quant_svcj.pyx", "svcj.c"],
        include_dirs=["."], # Look for svcj.h in root
        extra_compile_args=["-O3", "-fopenmp"] if sys.platform != "win32" else ["/O2", "/openmp"],
        extra_link_args=["-fopenmp"] if sys.platform != "win32" else [],
    )
]

setup(
    name="QuantSVCJ",
    version="6.0.1",
    description="High-performance SVCJ Engine",
    py_modules=["quantsvcj"],
    ext_modules=ext_modules,
    # Use the custom build class
    cmdclass={'build_ext': CustomBuildExt},
    # setup_requires ensures these are present before setup runs fully
    setup_requires=["numpy", "cython"],
    install_requires=["numpy", "pandas", "cython"],
    zip_safe=False,
)