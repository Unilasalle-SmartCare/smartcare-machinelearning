import os
import tensorflow as tf
from keras.models import load_model
from .ConfigEnum import ConfigEnum

class Brain:

    def __init__(self, mode=0):
        self.config = ConfigEnum()
        self.modeSwitcher = {
            0: self.config.get("CNN_DEFAULT_MODEL")
        }
        self.model = None
        self.init(self.modeSwitcher[mode])
    def init(self, mode):
        tf.get_logger().setLevel('ERROR')
        tf.autograph.set_verbosity(0)
        self.model = load_model(mode)

    def predict(self, image):
        prediction = self.model.predict(image)
        result = {
            True: "Wandering",
            False: "Not Wandering"
        }
        prediction = [prediction > 0.5]
        return result[prediction[0][0][0]]