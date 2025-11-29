from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        "svcj_wrapper", 
        sources=["svcj_wrapper.pyx", "svcjmath.c"], 
        include_dirs=[numpy.get_include()],
        extra_compile_args=["-O3"]
    )
]

setup(
    name='SVCJFastEstimator',
    ext_modules=cythonize(extensions),
    version='1.0'
)