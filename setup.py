import numpy
from setuptools import setup, Extension
from Cython.Build import cythonize
import os

# Define the C Extension
ext_modules = [
    Extension(
        name="_quant_svcj",  # The compiled module name
        sources=["_quant_svcj.pyx", "svcj.c"], # Source files
        include_dirs=[".", numpy.get_include()], # Look in root and numpy
        # Optimization flags
        extra_compile_args=["-O3", "-fopenmp"],
        extra_link_args=["-fopenmp"],
    )
]

setup(
    name="QuantSVCJ",
    version="5.0.1",
    py_modules=["quantsvcj"], # This installs quantsvcj.py
    ext_modules=cythonize(ext_modules),
    zip_safe=False,
)