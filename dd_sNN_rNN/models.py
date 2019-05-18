import sklearn
import sklearn.metrics
import time
import os
import tempfile
import numpy as np
import keras.optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
import np_utils

computed_metrics = ['accuracy', 'mean_squared_error']

def get_callbacks(model_fname):
    return [ModelCheckpoint(model_fname)]

def get_loss(regression):
    if regression:
        return 'mean_squared_error'
    else:
        return 'categorical_crossentropy'

def get_optimizer(regression, nb_layers, lr_mult=1):
    if regression:
        return keras.optimizers.RMSprop(lr=0.001 / (1.5 * nb_layers) * lr_mult)
    else:
        return keras.optimizers.RMSprop(lr=0.001 * lr_mult)


def generate_conv_net_base(
        input_shape, nb_classes,
        nb_dense=128, nb_filters=32, nb_layers=1, lr_mult=1,
        kernel_size=(3, 3), stride=(1, 1),
        regression=False):
    assert nb_layers >= 0
    assert nb_layers <= 3
    model = Sequential()
    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                            border_mode='same',
                            input_shape=input_shape,
                            subsample=stride,
                            activation='relu'))
    model.add(Convolution2D(nb_filters, 3, 3, border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    if nb_layers > 1:
        model.add(Convolution2D(nb_filters * 2, 3, 3, border_mode='same', activation='relu'))
        model.add(Convolution2D(nb_filters * 2, 3, 3, border_mode='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

    if nb_layers > 2:
        model.add(Convolution2D(nb_filters * 4, 3, 3, border_mode='same', activation='relu'))
        model.add(Convolution2D(nb_filters * 4, 3, 3, border_mode='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(nb_dense, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    if not regression:
        model.add(Activation('softmax'))

    loss = get_loss(regression)
    model.compile(loss=loss,
                  optimizer=get_optimizer(regression, nb_layers, lr_mult=lr_mult),
                  metrics=computed_metrics)
    return model


def generate_conv_net(input_shape, nb_classes,
                      nb_dense=128, nb_filters=32, nb_layers=1, lr_mult=1,
                      regression=False):
    return generate_conv_net_base(
            input_shape, nb_classes,
            nb_dense=nb_dense, nb_filters=nb_filters, nb_layers=nb_layers, lr_mult=lr_mult,
            regression=regression)


def generate_vgg16_conv(input_shape, full_16=False, dropout=True):
    border_mode = 'same'
    model = Sequential()
    model.add(Convolution2D(64, 3, 3, activation='relu', border_mode=border_mode,
                            input_shape=input_shape))
    model.add(Convolution2D(64, 3, 3, activation='relu', border_mode=border_mode))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    if dropout:
        model.add(Dropout(0.25))

    model.add(Convolution2D(128, 3, 3, activation='relu', border_mode=border_mode))
    model.add(Convolution2D(128, 3, 3, activation='relu', border_mode=border_mode))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    if dropout:
        model.add(Dropout(0.25))

    model.add(Convolution2D(256, 3, 3, activation='relu', border_mode=border_mode))
    model.add(Convolution2D(256, 3, 3, activation='relu', border_mode=border_mode))
    model.add(Convolution2D(256, 3, 3, activation='relu', border_mode=border_mode))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    if dropout:
        model.add(Dropout(0.25))

    if full_16:
        for i in xrange(2):
            model.add(Convolution2D(512, 3, 3, activation='relu', border_mode=border_mode))
            model.add(Convolution2D(512, 3, 3, activation='relu', border_mode=border_mode))
            model.add(Convolution2D(512, 3, 3, activation='relu', border_mode=border_mode))
            model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
            if dropout:
                model.add(Dropout(0.25))

    return model


def generate_vgg16(input_shape, nb_classes, full_16=False, regression=False):
    model = generate_vgg16_conv(input_shape, full_16=full_16, dropout=True)

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))

    if not regression:
        model.add(Activation('softmax'))

    loss = get_loss(regression)
    model.compile(loss=loss,
                  optimizer=get_optimizer(regression, 8 + full_16 * 8),
                  metrics=computed_metrics)
    return model


