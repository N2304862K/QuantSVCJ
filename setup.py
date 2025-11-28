import os
from setuptools import setup, Extension, find_packages

# --- ROBUST BUILD STRATEGY ---
# Allows 'pip install' to read metadata before numpy/cython are installed.
try:
    from Cython.Build import cythonize
    import numpy
    HAS_BUILD_DEPS = True
except ImportError:
    HAS_BUILD_DEPS = False

# Define Extension only if dependencies exist, otherwise empty (filled later by pip)
if HAS_BUILD_DEPS:
    extensions = [
        Extension(
            "quantsvcj._quant_svcj",
            sources=["quantsvcj/_quant_svcj.pyx", "quantsvcj/svcj.c"],
            include_dirs=[numpy.get_include(), "quantsvcj"],
            extra_compile_args=["-O3", "-fopenmp"],
            extra_link_args=["-fopenmp"],
        )
    ]
    ext_modules = cythonize(extensions)
else:
    ext_modules = []

setup(
    name="QuantSVCJ",
    version="5.0.0",
    description="High-performance SVCJ Factor Engine",
    packages=find_packages(),
    # Critical for building without pyproject.toml:
    setup_requires=["numpy", "cython"], 
    install_requires=[
        "numpy", 
        "pandas", 
        "cython"
    ],
    ext_modules=ext_modules,
    include_package_data=True,
    zip_safe=False,
)