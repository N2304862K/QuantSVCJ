import numpy
from setuptools import setup, Extension
from Cython.Build import cythonize

ext_modules = [
    Extension(
        name="_quant_svcj",
        sources=["_quant_svcj.pyx", "svcj.c"],
        include_dirs=[".", numpy.get_include()],
        extra_compile_args=["-O3", "-fopenmp"],
        extra_link_args=["-fopenmp"],
    )
]

setup(
    name="QuantSVCJ",
    version="6.0.0",
    py_modules=["quantsvcj"],
    ext_modules=cythonize(ext_modules),
    zip_safe=False,
)