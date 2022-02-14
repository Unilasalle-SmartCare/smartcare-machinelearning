import os
import tensorflow as tf
from keras.models import load_model
from .Config import Config

class Brain:

    def __init__(self, mode=0):
        self.modeSwitcher = {
            0: Config("CNN_DEFAULT_MODEL")
        }
        self.model = None
        self.init(self.modeSwitcher[mode])
    def init(self, mode):
        tf.get_logger().setLevel('ERROR')
        tf.autograph.set_verbosity(Config("NORMALIZATION_ENABLED"))
        self.model = load_model(mode)

    def predict(self, image):
        prediction = self.model.predict(image)
        result = {
            True: Config("RESULT_MAP")["WANDER"],
            False: Config("RESULT_MAP")["NORMAL"]
        }
        prediction = [prediction > Config("RESULT_THRESHOLD")]
        return result[prediction[0][0][0]]

    def predict_profiler(self, image):
        tf.profiler.experimental.start(Config("LOGS_FOLDER"))
        self.model.predict(image)
        tf.profiler.experimental.stop()