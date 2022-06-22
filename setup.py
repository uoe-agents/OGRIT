import setuptools
from setuptools import setup

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setuptools.setup(name="grit-odr",
                 author="Anonymous Author",
                 version='0.1.0',
                 install_requires=requirements,
                 packages=setuptools.find_packages(exclude=['analysis', 'test', 'scripts'])
                 )
