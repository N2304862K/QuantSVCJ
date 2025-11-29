import sys
import numpy
from setuptools import setup, Extension
from Cython.Build import cythonize

# Detect OS for OpenMP flags
if sys.platform == "win32":
    compile_args = ["/O2", "/openmp"]
    link_args = []
else:
    compile_args = ["-O3", "-fopenmp"]
    link_args = ["-fopenmp"]

# Define Extension
ext = Extension(
    name="_quant_svcj",
    sources=["_quant_svcj.pyx", "svcj.c"],
    include_dirs=[".", numpy.get_include()],
    extra_compile_args=compile_args,
    extra_link_args=link_args,
)

setup(
    name="QuantSVCJ",
    version="6.0.2",
    py_modules=["quantsvcj"],
    ext_modules=cythonize([ext]),
    zip_safe=False,
)