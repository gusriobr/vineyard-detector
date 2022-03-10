import logging

import tensorflow as tf
from keras import layers
from keras.applications.resnet import ResNet50
from keras.applications.vgg19 import VGG19
from keras.layers import Resizing

logging.basicConfig(level=logging.INFO)

IMG_SIZE = 48

resize = Resizing(144, 144)
def build_binary_classifier(activation="relu", augmentation_layer=None, freeze_weights=True):
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = inputs
    # x = resize(x)
    if augmentation_layer is not None:
        x = augmentation_layer(x)
    # trained_model = EfficientNetB0(include_top=False, input_tensor=x, weights="imagenet", drop_connect_rate=0.4)
    # trained_model = ResNet50V2(include_top=False, input_tensor=x, weights="imagenet")
    trained_model = VGG19(include_top=False, input_tensor=x, weights="imagenet")

    # Freeze the pretrained weights
    if freeze_weights:
        trained_model.trainable = False

    # Rebuild top
    x = layers.GlobalAveragePooling2D(name="avg_pool")(trained_model.output)
    x = layers.BatchNormalization()(x)

    top_dropout_rate = 0.2
    x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    outputs = layers.Dense(1, activation=activation, name="pred")(x)  # binary classification

    # Compile
    model = tf.keras.Model(inputs, outputs, name="EfficientNet")

    return model


def build_binary_classifier_restNet(activation="relu", augmentation_layer=None, freeze_weights=True):
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = inputs
    # x = resize(inputs)
    if augmentation_layer is not None:
        x = augmentation_layer(x)
    # trained_model = EfficientNetB0(include_top=False, input_tensor=x, weights="imagenet", drop_connect_rate=0.4)
    # trained_model = ResNet50V2(include_top=False, input_tensor=x, weights="imagenet")
    trained_model = ResNet50(include_top=False, input_tensor=x, classes=1)

    # Freeze the pretrained weights
    if freeze_weights:
        trained_model.trainable = False

    # Rebuild top
    x = layers.GlobalAveragePooling2D(name="avg_pool")(trained_model.output)
    x = layers.BatchNormalization()(x)

    top_dropout_rate = 0.2
    x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    outputs = layers.Dense(1, activation=activation, name="pred")(x)  # binary classification

    # Compile
    model = tf.keras.Model(inputs, outputs, name="ResNet50")

    return model
