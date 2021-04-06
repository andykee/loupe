import re

from setuptools import setup

with open('loupe/__init__.py') as f:
    version = re.search("__version__ = '(.*?)'", f.read()).group(1)

setup(
    name='Loupe',
    version=version,
    install_requires=[
        'numpy',
        'scipy'
    ]
)