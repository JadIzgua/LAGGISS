# -*- coding: utf-8 -*-
"""
Description of the script
"""
# ----------------------------------------------------------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------------------------------------------------------
from keras import Input, Model
from keras.applications import VGG19
from keras.layers import concatenate, Conv2D, Dropout, Flatten, Dense, BatchNormalization
from keras.optimizers import SGD
import numpy as np
import talos as ta
import pandas as pd
import wrangle
import os

# ----------------------------------------------------------------------------------------------------------------------
# Functions definitions
# ----------------------------------------------------------------------------------------------------------------------
def model(train_data, train_label, val_data, val_label, params):
    """
    """
    img_size = 224
    vgg_feature_extractor = VGG19(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))
    for layer in vgg_feature_extractor.layers[:17]:
        layer.trainable = False

    img_a = Input(shape=(img_size, img_size, 3), name="left_image")
    img_b = Input(shape=(img_size, img_size, 3), name="right_image")

    out_a = vgg_feature_extractor(img_a)
    out_b = vgg_feature_extractor(img_b)

    concat = concatenate([out_a, out_b])

    x = Dense(params['dense_1'], (3, 3), activation='relu', padding='same', name="Dense_1")(concat)
    x = Dropout(params['drop_1'], name="drop_1")(x)
    x = Dense(params['dense_2'], (3, 3), activation='relu', padding='same', name="Dense_2")(x)
    x = Dropout(params['drop2_1'], name="drop_2")(x)
    x = Dense(params['dense_3'], (3, 3), activation='relu', padding='same', name="Dense_3")(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)
    x = Dense(2, activation='softmax', name="Final_dense")(x)

    model = Model([img_a, img_b], x)

    sgd = SGD(lr=params['lr'], decay=params['decay'], momentum=params['momentum'], nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    out = model.fit(x=train_data, y=train_label, batch_size=params['batch_size'], epochs=params['epochs'],
                    validation_data=[val_data, val_label])

    return out, model


if __name__ == "__main__":
    # ------------------------------------------------------------------------------------------------------------------
    # Variables initialization
    # ------------------------------------------------------------------------------------------------------------------
    # Hyperparameters dictionnary
    p = {'dense_1': [64, 128, 256, 512],
         'dense_2': [64, 128, 256, 512],
         'dense_3': [64, 128, 256, 512],
         'drop_1': (0, 0.40, 10),
         'drop_2': (0, 0.40, 10),
         'lr': [1e-3, 1e-4, 1e-5, 1e-6],
         'decay': [1e-3, 1e-4, 1e-5, 1e-6],
         'momentum': (0, 0.9, 10),
         'batch_size': [16, 32],
         'epochs': [50, 100]}

    #  Loading data
    data_folder = r"D:\Guillaume\Ottawa\Data\Comparisons_npy\08_13"
    train_left = np.load(os.path.join(data_folder, "train", "train_left_224.npy"))
    train_right = np.load(os.path.join(data_folder, "train", "train_right_224.npy"))
    train_label = np.load(os.path.join(data_folder, "train", "train_labels_224.npy"))
    train_data = np.array([train_left, train_right])

    train_data, train_label, val_data, val_label = wrangle.array_split(train_data, train_label, .2)

    # ------------------------------------------------------------------------------------------------------------------
    # Run functions
    # ------------------------------------------------------------------------------------------------------------------
    h = ta.Scan(x=train_data,
                y=train_label,
                x_val=val_data,
                y_val=val_label,
                params=p,
                model=model,
                experiment_no='1')
