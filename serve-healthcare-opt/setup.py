from setuptools import setup, find_packages
from os import path
from io import open

here = path.abspath(path.dirname(__file__))

setup(
    name="ensemble_profiler",
  version='0.0.1',
  packages=find_packages(exclude=['contrib', 'docs', 'tests']),
  python_requires='>=3, <4',
  install_requires=['ray==0.9.0.dev', 'uvloop','torch','torchvision']
)
