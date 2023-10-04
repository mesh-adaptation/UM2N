try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


setup(
    name='warpmesh',
    use_scm_version=True,
    setup_requires=['setuptools_scm'],
    version='1.0',
    description='PDE mesh r-adaptation tool',
    author='Chunyang Wang',
    packages=['warpmesh'],
)
