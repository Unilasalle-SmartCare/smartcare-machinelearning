from setuptools import setup, find_packages
import codecs
import os

VERSION = '0.0.21'
DESCRIPTION = 'Elderly wandering prediction AI'
LONG_DESCRIPTION = 'Originally from the paper ''Convolutional Neural Network for Elderly Wandering Prediction in indoor scenarios.'' An AI to detect elderly wandering by movementation in indoor scenarios.'

# Setting up
setup(
    name="smartcare",
    version=VERSION,
    author="Rafael Faustini",
    author_email="<contato@rafaelfaustini.com.br>",
    description=DESCRIPTION,
    package_data={'smartcare': ['model/*']},
    include_package_data=True,
    long_description_content_type="text/markdown",
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    install_requires=['tensorflow', 'keras', 'pillow', 'tensorboard_plugin_profile'],
    keywords=['machinelearning', 'paper', 'smartcare', 'cnn', 'artificial inteligence', 'alzheimer', 'health'],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)