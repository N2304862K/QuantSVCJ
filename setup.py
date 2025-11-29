import numpy
from setuptools import setup, Extension
from Cython.Build import cythonize
import os

# Define the Extension
extensions = [
    Extension(
        name="_svcj_impl", 
        sources=["_svcj_impl.pyx", "svcj.c"],
        include_dirs=[numpy.get_include(), "."],
        extra_compile_args=["-O3", "-fopenmp", "-fPIC"],
        extra_link_args=["-fopenmp"],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
    )
]

setup(
    name="quantsvcj",
    version="2.0.0",
    description="QuantSVCJ: Flat SVCJ Factor Engine",
    py_modules=["quantsvcj"], # This includes quantsvcj.py
    ext_modules=cythonize(extensions, compiler_directives={'language_level': "3"}),
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0"
    ],
    zip_safe=False,
)