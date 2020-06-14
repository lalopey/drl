#!/usr/bin/env python
# Package for Udacity Deep Reinforcement Learning Program

from distutils.core import setup

setup(name='DRL Tools',
      version='0.1',
      author='Eduardo Peynetti',
      author_email='lalopey@gmail.com',
      url='https://github.com/lalopey/reinforcement-learning',
      packages=['drltools',
                'drltools.agent',
                'drltools.model',
                'drltools.utils'], requires=['torch', 'unityagents']
     )
