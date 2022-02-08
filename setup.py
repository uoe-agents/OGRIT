from setuptools import setup

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(name="grit-odr",
      author="Cillian Brewitt",
      version='1.0.0',
      install_requires=requirements)
