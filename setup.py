import sys
import numpy
from setuptools import setup, Extension
from Cython.Build import cythonize

# OpenMP / OS Specific Flags
if sys.platform.startswith("win"):
    compile_args = ["/openmp", "/O2"]
    link_args = []
else:
    # Linux/Mac Optimization flags
    compile_args = ["-fopenmp", "-O3", "-ffast-math"]
    link_args = ["-fopenmp"]

extensions = [
    Extension(
        "svcj_wrapper",
        sources=["svcj_wrapper.pyx", "svcj_kernel.c"],
        include_dirs=[numpy.get_include(), "."],
        extra_compile_args=compile_args,
        extra_link_args=link_args,
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
    )
]

setup(
    name="SVCJ_Factor_Engine_Pro",
    ext_modules=cythonize(extensions, compiler_directives={'language_level': "3"}),
    zip_safe=False,
)