# -*- coding: utf-8 -*-
"""
Script used to define the comparisons model and trained it.
"""
# ----------------------------------------------------------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------------------------------------------------------
import os
import numpy as np

from keras import Input, Model
from keras.applications import VGG19
from keras.layers import concatenate, Conv2D, Dropout, Flatten, Dense, BatchNormalization, Activation
from tensorflow.keras.layers.experimental.preprocessing import RandomTranslation, RandomFlip, RandomRotation, RandomZoom, RandomHeight, RandomWidth 
from tensorflow.keras.optimizers import SGD, Adam

from Class_training import simple_training, k_fold
from utils_class import shuffle_unison_arrays


data_augmentation = Sequential([
                                RandomFlip("horizontal_and_vertical"),
                                RandomRotation((0,0.5), fill_mode='constant')
])



run = True
# ----------------------------------------------------------------------------------------------------------------------
# Functions definitions
# ----------------------------------------------------------------------------------------------------------------------
def comparisons_model(img_size, weigths=None):
    """
    Create comparisons network which reproduce the choice in an images duel.

    :param img_size: size of input images during training
    :type img_size: tuple(int)
    :param weights: path to the weights use for initialization
    :type weights: str
    :return: ranking comparisons model
    :rtype: keras.Model
    """
    vgg_feature_extractor = VGG19(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))

    # Fine tuning by freezing the last 4 convolutional layers of VGG19 (last block)
    for layer in vgg_feature_extractor.layers[:-4]:
        layer.trainable = False

    # Definition of the 2 inputs
    img_a = Input(shape=(img_size, img_size, 3), name="left_image")
    img_b = Input(shape=(img_size, img_size, 3), name="right_image")

    pre_img_a = data_augmentation(img_a)
    pre_img_b = data_augmentation(img_b)

    out_a = vgg_feature_extractor(pre_img_a)
    out_b = vgg_feature_extractor(pre_img_b)

    # Concatenation of the inputs
    concat = concatenate([out_a, out_b])

    # Add convolution layers on top
    x = Conv2D(1024, (3, 3), padding='same', name="Conv_1")(concat)
    x = BatchNormalization()(x)
    x = Activation('relu', name='Activation_1')(x)
    x = Dropout(0.66, name="Drop_1")(x)
    x = Conv2D(512, (3, 3), padding='same', name="Conv_2")(x)
    x = BatchNormalization()(x)
    x = Activation('relu', name='Activation_2')(x)
    x = Conv2D(256, (3, 3), padding='same', name="Conv_3")(x)
    x = BatchNormalization()(x)
    x = Activation('relu', name='Activation_3')(x)
    x = Conv2D(128, (3, 3), padding='same', name="Conv_4")(x)
    x = BatchNormalization()(x)
    x = Activation('relu', name='Activation_4')(x)
    x = Dropout(0.5, name="Drop_2")(x)
    x = Flatten()(x)
    x = Dense(2, activation='softmax', name="Final_dense")(x)

    classification_model = Model([img_a, img_b], x)
    if weigths:
        classification_model.load_weights(weigths)
    #sgd = SGD(learning_rate=1e-5, decay=1e-6, momentum=0.3, nesterov=True)
    adam = Adam(learning_rate=1e-5)
    classification_model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    return classification_model

if run:
    # ------------------------------------------------------------------------------------------------------------------
    # Variables initialization
    # ------------------------------------------------------------------------------------------------------------------
    width = 224
    save_folder = r"C:\PythonData\LAGGISS\Save_folder"
    data_left = np.load(os.path.join(save_folder, "train", "train_left_224.npy"))
    data_right = np.load(os.path.join(save_folder, "train", "train_right_224.npy"))
    data_label = np.load(os.path.join(save_folder, "train", "train_labels_224.npy"))
    # ------------------------------------------------------------------------------------------------------------------
    # Run functions
    # ------------------------------------------------------------------------------------------------------------------
    # Build npy
    # trainLeft, trainRight, train_label = preprocessing_duels(csv_path, width, height, img_dir, save_folder, 0.1)

    # Shuffling data
    data_left_shuffled, data_right_shuffled, data_label_shuffled = shuffle_unison_arrays([data_left, data_right, data_label])
    print("Starting")

    # Simple training
    folder_path = r"C:\PythonData\LAGGISS\Comp_model1"
    simple_training(data_left_shuffled, data_right_shuffled, data_label_shuffled, comparisons_model, [width], folder_path, val_split=0.2, epochs=15, batch_size=32)

    print("done")
    
'''
    # K fold training
    folder_path = r"C:\PythonData\LAGGISS\Comp_model_k_fold"
    k_fold(data_left, data_right, data_label, 5, comparisons_model, [width], folder_path)

    # Evaluation on test data
    save_folder = r"C:\PythonData\LAGGISS\Save_folder"
    model_path = os.path.join(save_folder, "k-fold", "fitted_comparisons_k5_clear_session.h5")
    evaluation_test_set(model_path, save_folder, mode="comparisons")
'''
