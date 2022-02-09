#!/usr/bin/env python

from setuptools import setup, find_packages
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(name='RFSM',
      version='0.0.1',
      description='🌎 Ejecution RFSM.',
      long_description=long_description,
      long_description_content_type='text/markdown',
      author='Salvador Navas',
      author_email='navass@unican.es',
      url = 'https://github.com/IHCantabria/RFSM',
      packages = ['RFSM_python'],
      include_package_data = True,
      python_requires='>=3.7, <4',
      install_requires=[
          'numpy',
          'pandas',
          'scipy',
          'datetime',
          'matplotlib',
          'pyyaml',
      ],
      extras_require={'plotting': ['matplotlib>=2.2.0', 'jupyter','jupyterlab']}
     )
     
