import sys
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

# Custom build class to defer numpy import until build time
class CustomBuildExt(build_ext):
    def finalize_options(self):
        build_ext.finalize_options(self)
        # Prevent numpy from being imported before it's installed
        import builtins
        builtins.__NUMPY_SETUP__ = False
        import numpy
        self.include_dirs.append(numpy.get_include())

# Define the Extension
# Note: Sources are all in the root directory
ext_modules = [
    Extension(
        "_quant_svcj", # Name of the compiled module
        sources=["_quant_svcj.pyx", "svcj.c"],
        include_dirs=["."], # Look in root for .h files
        extra_compile_args=["-O3", "-fopenmp"] if sys.platform != "win32" else ["/O2", "/openmp"],
        extra_link_args=["-fopenmp"] if sys.platform != "win32" else [],
    )
]

setup(
    name="QuantSVCJ",
    version="5.0.0",
    description="Flat-structure SVCJ Factor Engine",
    py_modules=["quantsvcj"], # This includes quantsvcj.py
    ext_modules=ext_modules,
    cmdclass={'build_ext': CustomBuildExt},
    setup_requires=["numpy", "cython"],
    install_requires=["numpy", "pandas", "cython"],
    zip_safe=False,
)