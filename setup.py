from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy
import sys

# Compiler flags for optimization and OpenMP
if sys.platform.startswith("win"):
    compile_args = ["/openmp", "/O2"]
else:
    compile_args = ["-fopenmp", "-O3"]
    
link_args = compile_args

extensions = [
    Extension(
        "svcj_wrapper",
        sources=["svcj_wrapper.pyx", "svcj.c"],
        include_dirs=[numpy.get_include(), "."],
        extra_compile_args=compile_args,
        extra_link_args=link_args,
    )
]

setup(
    name="SVCJ_Factor_Engine",
    ext_modules=cythonize(extensions, compiler_directives={
        'language_level': "3",
        'boundscheck': False,
        'wraparound': False,
        'cdivision': True
    }),
)