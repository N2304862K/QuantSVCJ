import sys
import os
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

# Custom builder to delay numpy import until pip has installed it
class BuildExt(build_ext):
    def run(self):
        import numpy
        self.include_dirs.append(numpy.get_include())
        super().run()

# Define the Extension
ext_modules = [
    Extension(
        "_svcj_impl",
        sources=["_svcj_impl.pyx", "svcj.c"],
        include_dirs=["."],
        extra_compile_args=["-O3", "-fopenmp"],
        extra_link_args=["-fopenmp"],
    )
]

# Cythonize if available, else rely on pre-generated C (if distributed that way)
try:
    from Cython.Build import cythonize
    ext_modules = cythonize(ext_modules, compiler_directives={'language_level': "3"})
except ImportError:
    pass

setup(
    name="quantsvcj",
    version="1.0.0",
    description="QuantSVCJ: High-Performance Factor Engine",
    py_modules=["quantsvcj"],
    cmdclass={'build_ext': BuildExt},
    ext_modules=ext_modules,
    setup_requires=["numpy>=1.20.0", "cython>=0.29.0", "setuptools>=40.0.0"],
    install_requires=["numpy>=1.20.0", "pandas>=1.3.0", "cython>=0.29.0"],
    zip_safe=False,
)