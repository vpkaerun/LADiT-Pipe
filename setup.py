
from setuptools import setup, find_packages

setup(
    name="ladit_pipe",
    version="1.1.0",
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'ladit-pipe = ladit_pipe.main:main',
        ],
    },
)
