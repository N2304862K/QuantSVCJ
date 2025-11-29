from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

ext = Extension(
    "quantsvcj._quant_svcj",
    sources=["_quant_svcj.pyx", "svcj.c"],
    include_dirs=[numpy.get_include(), "."],
    extra_compile_args=["-O3", "-fopenmp"],
    extra_link_args=["-fopenmp"]
)

setup(
    name="QuantSVCJ",
    version="5.0",
    packages=["quantsvcj"],
    ext_modules=cythonize([ext]),
    install_requires=["numpy", "pandas", "cython"],
)