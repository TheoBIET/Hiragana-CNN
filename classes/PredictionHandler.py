import sys
sys.path.append("..")
from PIL import Image, ImageOps
from tensorflow.keras.models import load_model
import skimage.io
import numpy as np

from utils.constants import HIRAGANA

TARGET_SIZE = (48, 48)
TARGET_SHAPE = (48, 48, 1)
MODEL_SHAPE = (1, 48, 48, 1)
HIRA = [' A',
 ' I',
 ' U',
 ' E',
 ' O',
 'KA',
 'KI',
 'KU',
 'KE',
 'KO',
 'SA',
 'SI',
 'SU',
 'SE',
 'SO',
 'TA',
 'TI',
 'TU',
 'TE',
 'TO',
 'NA',
 'NI',
 'NU',
 'NE',
 'NO',
 'HA',
 'HI',
 'HU',
 'HE',
 'HO',
 'MA',
 'MI',
 'MU',
 'ME',
 'MO',
 'YA',
 'YU',
 'YO',
 'RA',
 'RI',
 'RU',
 'RE',
 'RO',
 'WA',
 'WO',
 ' N',
 ',,',
 ',0']

model = load_model('hiragana_model.h5')


class PredictionHandler():
    def __init__(self, image):
        self.model = model
        self.prediction_prob = None
        self.image = Image.open(image.stream).convert('L')
        self.image = self.__invert()
        self.image_array = self.__image_to_array()
        self.image_array = self.__resize()
        self.image_array = self.__reshape()
        self.image_array = self.__normalize()

    def make_prediction(self):
        self.prediction_prob = self.model.predict(self.image_array.reshape(MODEL_SHAPE))
        prediction = self.prediction_prob * 100
        return [(HIRA[i], prediction[0][i]) for i in range(len(HIRAGANA)) if prediction[0][i] > 0.5]

    def __invert(self):
        return ImageOps.invert(self.image)

    def __image_to_array(self):
        return np.array(self.image, dtype=np.float32)

    def __resize(self):
        return skimage.transform.resize(self.image_array, TARGET_SIZE)

    def __reshape(self):
        return self.image_array.reshape(TARGET_SHAPE)

    def __normalize(self):
        return self.image_array / 255
