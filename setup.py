from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext as _build_ext

# Custom build command to delay numpy import until installation ensures it exists
class build_ext(_build_ext):
    def finalize_options(self):
        _build_ext.finalize_options(self)
        # Prevent numpy from importing before pip installs it
        __builtins__.__NUMPY_SETUP__ = False
        import numpy
        self.include_dirs.append(numpy.get_include())

# Define the Extension
ext_modules = [
    Extension(
        "_svcj_impl",
        sources=["_svcj_impl.pyx", "svcj.c"],
        include_dirs=["."], # Include current dir for svcj.h
        extra_compile_args=["-O3", "-fopenmp"],
        extra_link_args=["-fopenmp"],
    )
]

setup(
    name="quantsvcj",
    version="1.0.0",
    description="Flat SVCJ Estimation Engine",
    py_modules=["quantsvcj"], # Include the python portal file
    ext_modules=ext_modules,
    cmdclass={'build_ext': build_ext},
    setup_requires=['numpy>=1.20.0', 'cython>=0.29.0', 'setuptools>=40.0'],
    install_requires=['numpy>=1.20.0', 'pandas>=1.3.0'],
    zip_safe=False,
)