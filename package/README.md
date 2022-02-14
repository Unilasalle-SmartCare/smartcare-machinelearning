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
imageUtil = Image.fromPath(path) # 2)
```

As you can see in case one, you can pass an image in base64 or just like _2)_, giving the path of the image in disk. After instanciating the Image class, all the pre processing is done automatically. To obtain the processed image you can use the method as it follows:

```python
imageUtil.get() ## 3)
```

The get method _3)_ returns the Image Grayscaled, resized, normalized in numpy array format ready to go in the network model.

### Brain

The Brain class contains the machine learning logic, from opening the model until passing the path image that later on generates the prediction of wandering.

```python
from smartcare.Brain import Brain

aiModel = Brain() # 4)
aiModel.predict(processedImage) # 5)
```

In item _4)_, the machine learning model is loaded and prepared for the predictions. For the predictions you use the predict method seen in _5)_. The predict method receives as parameter the processed image _3)_ and returns the prediction according to the RESULT_MAP config.

### Config

| Config                | Default                     | Description                                                                  |
| --------------------- | --------------------------- | ---------------------------------------------------------------------------- |
| IMAGE_HEIGHT          | 128                         | Target height for the image resize                                           |
| IMAGE_WIDTH           | 128                         | Target width for the image resize                                            |
| LOGS_FOLDER           | logs                        | Log folder for the profiler                                                  |
| RESULT_THRESHOLD      | 0.5                         | Sensitivity of what is considered wandering or not from the predicted result |
| RESULT_MAP            | { "WANDER": 1, "NORMAL": 0} | Values to be returned when WANDER or NORMAL prediction                       |
| VERBOSITY             | 1                           | Tensorflow prediction verbosity                                              |
| RESIZE_ENABLED        | True                        | Should image be resized on Image processing?                                 |
| NORMALIZATION_ENABLED | True                        | Should image be normalized on Image processing?                              |
| CNN_DEFAULT_MODEL     | (CNN TRAINED MODEL H5)      | H5 weights file, by the default it's a resource file in the package          |

#### Get

Obtaining config values is very easy, just by passing the config name in the class parameter

Get IMAGE_HEIGHT

```python
from smartcare.Config import Config
print(Config("IMAGE_HEIGHT"))

```

#### Set

Setting config files are also easy, using the set method

```python
from smartcare.Config import Config
Config.set("IMAGE_HEIGHT", 64)

```
