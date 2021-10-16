# Smartcare Machine Learning

It is proposed a way to detect wandering activities on alzheimer patients by their movimentation, collected in a non intrusive way.

## Paper

This resources come from the [paper](https://www.scitepress.org/PublicationsDetail.aspx?ID=QeUQHNkUUVM=&t=1) we've published on HEALTHINF 2021 conference

## Dataset

Generated data on the paper can be found [here](model/dados/dataset.json).<br>
The source-code of the generator [here](https://github.com/Unilasalle-SmartCare/smartcare-datagenerator)

## Model

Where the technical part of the deep learning is, the build and train of the convolutional neural network (CNN). Check it out by [clicking here](model/)

## Packages

A python package was made to abstract some of the pre processing and model run. For more details check it [here](package/)

## Server

It is planned in the future to have an api server with the model running (Flask). Check it out by [clicking here](server/)

## Releases

In order to make it easier to join the features and fixes developed in the project, it was grouped with releases, you can see them by [clicking here](https://github.com/Unilasalle-SmartCare/smartcare-machinelearning/releases).
