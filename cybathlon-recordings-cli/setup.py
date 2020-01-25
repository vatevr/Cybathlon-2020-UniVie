#!/usr/bin/env python

from distutils.core import setup

requirements = open('requirements.txt', 'r').read().strip().split('\n')

setup(
	name='cybathlon-cybathlon-recordings-cli',
	version='0.1.0',
	description='CLI interface for Cybathlon recordings API',
	author='Hamlet Mkrtchyan',
	author_email='hamlet.mkrtchyan@protonmail.ch',
	packages=['cybathlon-recordings-cli'],
	install_requires=requirements,
	entry_points={  # Optional
        'console_scripts': [
            'cybathlon-recordings-cli=cybathlon-cli.__main__:main',
        ],
    }
)