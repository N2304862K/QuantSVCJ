from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

ext = Extension(
    "quantsvcj._wrapper",
    sources=["quantsvcj/_wrapper.pyx", "src/svcj_engine.c"],
    include_dirs=[numpy.get_include(), "src"],
    extra_compile_args=["-O3", "-fopenmp"],
    extra_link_args=["-fopenmp"]
)

setup(
    name="QuantSVCJ",
    version="2.0",
    packages=["quantsvcj"],
    ext_modules=cythonize([ext]),
    install_requires=["numpy", "pandas", "cython"],
)