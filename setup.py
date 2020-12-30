from distutils.core import setup
from setuptools import find_packages


setup(
    name='qc',
    version='0.1',
    description='Quantum Chemistry with Conditional Wave Functions',
    author='Marcin Kuropatwińsk',
    author_email='marcin@talking2rabbit.com',
    url='https://www.marcinkuropatwinski.pl',
    package_dir={'': 'src'},
    packages=['qc','Praktyki'],
)

