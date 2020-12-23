from distutils.core import setup
from setuptools import find_packages

setup(
    name='qc',
    version='0.1',
    description='Quantum Chemistry with Conditional Wave Functions',
    author='Marcin Kuropatwińsk',
    author_email='marcin@talking2rabbit.com',
    url='https://www.marcinkuropatwinski.pl',
    packages=find_packages('src'),
    package_dir={'': 'src'},
)
