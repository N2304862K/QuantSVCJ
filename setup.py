import os
import sys
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

# --- Custom Builder to Handle Numpy Dependency ---
class BuildExt(build_ext):
    def finalize_options(self):
        super().finalize_options()
        # Prevent numpy from being imported before it is installed
        __builtins__.__NUMPY_SETUP__ = False
        import numpy
        self.include_dirs.append(numpy.get_include())

# --- Extension Definition ---
ext_modules = [
    Extension(
        "_svcj_impl",
        sources=["_svcj_impl.pyx", "svcj.c"],
        include_dirs=["."],
        extra_compile_args=["-O3", "-fopenmp"],
        extra_link_args=["-fopenmp"],
    )
]

try:
    from Cython.Build import cythonize
    ext_modules = cythonize(ext_modules, compiler_directives={'language_level': "3"})
except ImportError:
    pass # Fallback if Cython not yet installed, setup_requires will handle it

setup(
    name="quantsvcj",
    version="1.0.0",
    description="QuantSVCJ Factor Engine",
    py_modules=["quantsvcj"], # This exposes the python file in root
    cmdclass={'build_ext': BuildExt},
    ext_modules=ext_modules,
    setup_requires=["numpy", "cython"],
    install_requires=["numpy", "pandas", "cython"],
    zip_safe=False,
)