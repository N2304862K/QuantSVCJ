from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy
import sys

# Standard OpenMP flags for Parallel Execution
if sys.platform.startswith("win"):
    omp_args = ["/openmp"]
else:
    omp_args = ["-fopenmp"]

extensions = [
    Extension(
        "svcj_wrapper",
        sources=["svcj_wrapper.pyx", "svcj.c"],
        include_dirs=[numpy.get_include(), "."],
        extra_compile_args=omp_args,
        extra_link_args=omp_args,
    )
]

setup(
    name="SVCJ_Factor_Engine",
    ext_modules=cythonize(extensions, compiler_directives={'language_level': "3"}),
)