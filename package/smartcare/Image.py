import base64
import numpy as np
from PIL import Image as PilImage
from io import BytesIO
from .ConfigEnum import ConfigEnum


class Image:
    def __init__(self, image=None, path=""):
        self.config = ConfigEnum()
        if image: 
            self.image = image
        if path:
            self.image = self.seek(path)

        self.process()

    def get(self):
        return self.image

    
    def decode(self, image):
        image_bytes = base64.b64decode(image)
        image = BytesIO(image_bytes)
        return PilImage.open(image).convert('L')

    def encode(self, image):
        image = base64.b64encode(image)
        return image
    

    def resize(self,image):
        image = np.uint8(image)
        image = PilImage.fromarray(image).resize((self.config.get("IMAGE_WIDTH"), self.config.get("IMAGE_HEIGHT")))
        return np.array(image)

    def normalize(self, image):
        return image / 255

    def process(self):
        image = self.decode(self.image)
        image = np.array(image)
        image = self.resize(image)
        image = self.normalize(image)
        image = image[np.newaxis, ..., np.newaxis]
        print(image.shape)
        self.image = image
        return image

    def seek(self, path):
        image = PilImage.open(path).convert('L')
        im_file = BytesIO()
        image.save(im_file, format="JPEG")
        return self.encode(im_file.getvalue())

    