import numpy as np
import pydicom
import cv2
import tensorflow as tf
from math import ceil, floor
from tqdm import tqdm
import matplotlib.pyplot as plt
from math import ceil, floor
import keras
import keras.backend as K
from keras.callbacks import Callback, ModelCheckpoint
from keras.layers import Dense, Flatten, Dropout
from keras.models import Model, load_model
from keras.utils import Sequence
from keras.losses import binary_crossentropy
from keras.optimizers import Adam
import efficientnet.keras as efn
import joblib
import warnings

warnings.filterwarnings('ignore')

HEIGHT = 256
WIDTH = 256
CHANNELS = 3
SHAPE = (HEIGHT, WIDTH, CHANNELS)
path_test_img = 'dicom/'


def get_corrected_bsb_window(dcm, window_center, window_width):
    # ------ Correct Dicom Image ------------#
    if (dcm.BitsStored == 12) and (dcm.PixelRepresentation == 0) and (int(dcm.RescaleIntercept) > -100):
        x = dcm.pixel_array + 1000
        px_mode = 4096
        x[x >= px_mode] = x[x >= px_mode] - px_mode
        dcm.PixelData = x.tobytes()
        dcm.RescaleIntercept = -1000

    # ------ Windowing ----------------------#
    img = dcm.pixel_array * dcm.RescaleSlope + dcm.RescaleIntercept
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    img = np.clip(img, img_min, img_max)
    return img


def get_rgb_image(img):
    brain_img = get_corrected_bsb_window(img, 40, 80)
    subdural_img = get_corrected_bsb_window(img, 80, 200)
    soft_img = get_corrected_bsb_window(img, 40, 380)

    brain_img = (brain_img - 0) / 80
    subdural_img = (subdural_img - (-20)) / 200
    soft_img = (soft_img - (-150)) / 380
    bsb_img = np.array([brain_img, subdural_img, soft_img]).transpose(1, 2, 0)

    return bsb_img


def _read(path, desired_size=(WIDTH, HEIGHT)):
    dcm = pydicom.dcmread(path)

    try:
        img = get_rgb_image(dcm)
    except:
        img = np.zeros(desired_size)

    img = cv2.resize(img, desired_size[:2], interpolation=cv2.INTER_LINEAR)

    return img


class TestDataGenerator(keras.utils.Sequence):
    def __init__(self, ids, labels, batch_size=5, img_size=(512, 512), img_dir=path_test_img, \
                 *args, **kwargs):
        self.ids = ids
        self.labels = labels
        self.batch_size = batch_size
        self.img_size = img_size
        self.img_dir = img_dir
        self.on_epoch_end()

    def __len__(self):
        return int(ceil(len(self.ids) / self.batch_size))

    def __getitem__(self, index):
        indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        list_IDs_temp = [self.ids[k] for k in indices]
        X = self.__data_generation(list_IDs_temp)
        return X

    def on_epoch_end(self):
        self.indices = np.arange(len(self.ids))

    def __data_generation(self, list_IDs_temp):
        X = np.empty((self.batch_size, *self.img_size, 3))
        for i, ID in enumerate(list_IDs_temp):
            image = _read(self.img_dir + ID, self.img_size)
            X[i,] = image
        return X


def get_predictions(filename):
    data_generator_test = TestDataGenerator([filename], None,
                                            1,
                                            (WIDTH, HEIGHT),
                                            augment=False)

    K.clear_session()
    base_model = efn.EfficientNetB0(weights='imagenet', include_top=False,
                                    pooling='avg', input_shape=(HEIGHT, WIDTH, 3))
    x = base_model.output
    x = Dropout(0.125)(x)
    output_layer = Dense(6, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=output_layer)
    model.compile(optimizer=Adam(lr=0.0001), loss='binary_crossentropy', metrics=['acc'])
    model.load_weights('model_effnet_bo_087.h5')
    test_preds = model.predict_generator(data_generator_test, use_multiprocessing=True)
    K.clear_session()
    return test_preds
