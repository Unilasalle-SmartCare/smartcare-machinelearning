from pkg_resources import resource_filename, Requirement
class ConfigEnum():
    dict = {
        "IMAGE_HEIGHT": 128,
        "IMAGE_WIDTH": 128,
        "LOGS_FOLDER": "logs",
        "RESULT_THRESHOLD": 0.5,
        "RESULT_MAP": {
            "WANDER": 1,
            "NORMAL": 0
        },
        "VERBOSITY": 1,
        "RESIZE_ENABLED": True,
        "NORMALIZATION_ENABLED": True,
        "CNN_DEFAULT_MODEL": resource_filename(__name__, 'model/modelCNN-1605022224.h5')
    }

    def __new__(cls, key):
        return ConfigEnum.get(key)

    def get(key):
        return ConfigEnum.dict[key]
    
    def set(key, value):
        if key in ConfigEnum.dict.keys():
            ConfigEnum.dict[key] = value

Config = ConfigEnum
