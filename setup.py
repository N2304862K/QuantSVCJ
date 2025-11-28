from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        "quantsvcj._c_wrapper",
        sources=["quantsvcj/_c_wrapper.pyx", "src/svcj_core.c"],
        include_dirs=[numpy.get_include(), "src"],
        extra_compile_args=["-O3", "-fopenmp"], # Optimization flags
        extra_link_args=["-fopenmp"]
    )
]

setup(
    name="QuantSVCJ",
    version="1.0",
    packages=["quantsvcj"],
    ext_modules=cythonize(extensions),
    install_requires=["numpy", "pandas", "cython"],
    zip_safe=False,
)