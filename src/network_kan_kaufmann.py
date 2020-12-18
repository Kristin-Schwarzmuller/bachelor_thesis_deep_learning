from dataloader import DataLoader
import numpy as np
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Dropout, Activation, Flatten, concatenate
from keras.layers import Convolution2D, MaxPooling2D, Conv2D
from keras.utils import np_utils
from keras.layers import Merge
import warnings
import keras.optimizers
from keras import backend
import math
from keras.applications.resnet50 import ResNet50, conv_block, identity_block

from keras.layers import Input
from keras import layers
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import ZeroPadding2D
from keras.layers import AveragePooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import BatchNormalization
from keras.models import Model
from keras import backend as K
from keras.engine.topology import get_source_inputs
from keras.utils import layer_utils
from keras.utils.data_utils import get_file

import sys
import time

class Network:

    dl = DataLoader()#DataLoader which contains training data. This loader is provided in constructor
    model = Sequential()#Neural network

    def __init__(self, dloader):
        self.dl = dloader

  

    def train_with_euler(self, learning_rate=0.001):
        color_mean = self.dl.calculate_mean(self.dl.image_data)
        depth_mean = self.dl.calculate_mean(self.dl.depth_data)
        print("Normalizing data with mean for color: ")
        print(color_mean)
        print(" and depth: ")
        print(depth_mean)
        self.dl.zero_mean_normalize_color(color_mean)
        self.dl.zero_mean_normalize_depth(depth_mean)

        resultX = self.mergeRGBD(self.dl.image_data, self.dl.depth_data)
        resultY = self.calculate_relative_euler(self.dl.real_data, self.dl.real_result)
        print("Start Training with RGBD and relative Euler. Number of records in DataLoader is: " + str(
            np.size(self.dl.image_data, 0)))
        print("Learning rate:" + str(learning_rate))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # =============Funcitonal API==================================================
            i_image = keras.layers.Input(shape=(self.dl.DATA_RES_Y, self.dl.DATA_RES_X, 4))
            x = ZeroPadding2D((3, 3))(i_image)
            x = Conv2D(64, (7, 7), strides=(2, 2))(x)
            x = BatchNormalization(axis=-1)(x)
            x = Activation('relu')(x)
            x = MaxPooling2D((3, 3), strides=(2, 2))(x)

            x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
            x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
            x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

            x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
            x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
            x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
            x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

            x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
            x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
            x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
            x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
            x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
            x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

            x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
            x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
            x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

            x = AveragePooling2D((7, 7), padding='same')(x)

            x = Flatten()(x)
            o_image = Dense(512, activation='relu')(x)

            o_merged = keras.layers.Dense(256, activation='relu')(o_image)
            o_merged = keras.layers.Dense(64, activation='relu')(o_merged)
            o_merged = keras.layers.Dense(64, activation='linear')(o_merged)  
            o_merged = keras.layers.Dense(2, activation='linear')(o_merged)
            self.model = keras.models.Model(inputs=i_image, outputs=o_merged)

            print("Input shape of model:")
            print(self.model.input_shape)
            print("Output shape of model:")
            print(self.model.output_shape)

            adm = keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)            
            self.model.compile(loss='mean_squared_error', optimizer=adm, metrics=['accuracy'])

            X_train = resultX
            Y_train = resultY
            history = self.model.fit(X_train, Y_train, batch_size=32, nb_epoch=200, verbose=1)
            del history

    def save(self, file_name):
        self.model.save(filepath=file_name)

    