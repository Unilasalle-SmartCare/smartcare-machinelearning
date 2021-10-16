# Smartcare Python Package

## Installation

```
> pip install smartcare
```
Check more about it in the [pypi project](https://pypi.org/project/smartcare/)

## Usage

### Image

The responsability of this class is to abstract all the images pre processing.

```python
from smartcare.Image import Image

imageUtil = Image(img) # 1)
imageUtil = Image(path) # 2)
```

As you can see in case one, you can pass an image in base64 or just like *2)*, giving the path of the image in disk. After instanciating the Image class, all the pre processing is done automatically. To obtain the processed image you can use the method as it follows:

```python
imageUtil.get() ## 3)
```

The get method *3)* returns the Image Grayscaled, resized, normalized in numpy array format ready to go in the network model.

### Brain

The Brain class contains the machine learning logic, from opening the model until passing the path image that later on generates the prediction of wandering.

```python
from smartcare.Brain import Brain

aiModel = Brain() # 4)
aiModel.predict(processedImage) # 5)
```

In item *4)*, the machine learning model is loaded and prepared for the predictions. For the predictions you use the predict method seen in *5)*. The predict method receives as parameter the processed image *3)* and returns the prediction in strings "Wandering", "Not wandering".
