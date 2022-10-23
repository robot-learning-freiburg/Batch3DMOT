from setuptools import setup

setup(
   name='batch_3dmot',
   author='Martin Buechner',
   author_email='buechner@cs.uni-freiburg.de',
   packages=['batch_3dmot',
             'batch_3dmot.eval',
             'batch_3dmot.models',
             'batch_3dmot.preprocessing',
             'batch_3dmot.training',
             'batch_3dmot.utils'
             ],
   license='LICENSE.md',
   description='A batch solution to IMG/LiDAR/RADAR multi-object tracking using Graph Neural Networks.',
   long_description=open('README.md').read()
)