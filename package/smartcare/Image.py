import base64
import numpy as np
from PIL import Image as PilImage
from io import BytesIO
from .Config import Config


class Image:
    def __init__(self, image):
        if image: 
            self.image = image

        self.process()

    @classmethod
    def fromPath(cls, path):
        data = Image.seek(path)
        return cls(data)  

    def get(self):
        return self.image

    
    def decode(image):
        image_bytes = base64.b64decode(image)
        image = BytesIO(image_bytes)
        return PilImage.open(image).convert('L')

    def encode(image):
        image = base64.b64encode(image)
        return image
    

    def resize(image):
        image = np.uint8(image)
        image = PilImage.fromarray(image).resize((Config("IMAGE_WIDTH"), Config("IMAGE_HEIGHT")))
        return np.array(image)

    def normalize(image):
        return image / 255

    def process(self):
        image = Image.decode(self.image)
        image = np.array(image)
        if Config("RESIZE_ENABLED"):
            image = Image.resize(image)
        if Config("NORMALIZATION_ENABLED"):
            image = Image.normalize(image)
        image = image[np.newaxis, ..., np.newaxis]
        self.image = image
        return image

    def seek(path):
        image = PilImage.open(path).convert('L')
        im_file = BytesIO()
        image.save(im_file, format="JPEG")
        return Image.encode(im_file.getvalue())

    