import logging
import os

import joblib
import numpy as np
import tensorflow as tf
from keras import layers
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.models import load_model

import cfg
from data.model import DatasetDef
from evaluation.eval import evaluate_model_file
from models.transferlr import build_binary_classifier
from vineyard.data import dataset

logging.basicConfig(level=logging.INFO)

IMG_SIZE = 48
# https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning/

kwargs = {}

img_augmentation = Sequential([
    # layers.RandomRotation(factor=0.15),
    # layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
    # layers.RandomFlip(),
    layers.RandomContrast(factor=0.1),
    layers.RandomZoom(height_factor=(-0.2, 0.2))
], name="img_augmentation")


def fine_tunning(model_file):
    model = load_model(model_file)

    model.trainable = True
    model.compile(
        optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
    )
    return model


def save(object, file_path):
    import jsonpickle
    json = jsonpickle.encode(object)
    # save json content
    fo = open(file_path, "w")
    fo.write(json)
    fo.close()


def train(model, model_label, dataset_file, output_folder, epochs=200, batch_size=32):
    (x_train, y_train), (x_test, y_test) = dataset.load(dataset_file)
    # y_train = utils.to_categorical(y_train, 2) no needed for binary classifier
    # y_test = utils.to_categorical(y_test, 2)
    datagen = ImageDataGenerator(rescale=1. / 255,
                                 featurewise_center=True,
                                 featurewise_std_normalization=True,
                                 rotation_range=20,
                                 width_shift_range=0.2,
                                 height_shift_range=0.2,
                                 horizontal_flip=True,
                                 vertical_flip=True,
                                 brightness_range=(0.8, 1.2),
                                 channel_shift_range=30
                                 )
    # compute quantities required for featurewise normalization
    datagen.fit(x_train)

    train_gen = datagen.flow(x_train, y_train, batch_size=batch_size)
    val_gen = datagen.flow(x_test, y_test, batch_size=batch_size)
    # save model and imagen normalization parameters
    model_conf = {"mean": datagen.mean, "std": datagen.std}
    save(model_conf, os.path.join(output_folder, "{}.json".format(model_label)))

    model_path = os.path.join(output_folder, "{}.model".format(model_label))
    metric_to_monitor = "val_accuracy"
    min_delta = 0.0001
    k_callbacks = [
        ModelCheckpoint(model_path, monitor=metric_to_monitor, verbose=0, save_best_only=True, save_weights_only=False,
                        mode='max', save_freq="epoch"),
        ReduceLROnPlateau(monitor=metric_to_monitor, factor=0.1, patience=10, mode='auto', min_delta=min_delta,
                          cooldown=0, min_lr=1e-6),
        EarlyStopping(monitor=metric_to_monitor, patience=30, min_delta=min_delta)
    ]

    history = model.fit(
        train_gen,
        steps_per_epoch=len(y_train) // batch_size,
        epochs=epochs,
        validation_data=val_gen,
        validation_steps=len(y_test) // batch_size,
        callbacks=k_callbacks
    )
    #
    history_file = os.path.join(output_folder, "{}.hist".format(model_label))
    joblib.dump(history, history_file)

    return model_path


IMG_SIZE = 48
if __name__ == '__main__':
    dataset_file = "/media/gus/data/viticola/datasets/dataset_v2/dataset.npy"
    output_folder = cfg.results("iteration2")

    # model = build_binary_classifier()
    model_label = "vgg19"
    # model = ResNet.build(IMG_SIZE, IMG_SIZE, 3, 1, (3, 3, 3, 3), filters=(64, 64, 64, 64),
    #                      augmentation_layer=img_augmentation)
    # model = build_binary_classifier_restNet(augmentation_layer=img_augmentation, freeze_weights=False)
    model = build_binary_classifier(augmentation_layer=img_augmentation, freeze_weights=True)

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])

    model_path = train(model, model_label, dataset_file, output_folder, epochs=200, batch_size=96)

    model_conf_file = model_path.replace(".model", ".json")
    model_conf = DatasetDef.read(model_conf_file)
    # standarize test images images
    (x_train, y_train), (x_test, y_test) = dataset.load(dataset_file)
    x_tr = np.zeros(shape=x_test.shape, dtype=np.float32)
    x_tr = x_test * (1.0 / 255.0)
    x_tr -= model_conf["mean"]
    x_tr /= model_conf["std"]
    evaluate_model_file(model_path, x_tr, y_test)
    #
    # model = fine_tunning(model_path)
    # fine_tunned = train(model, model_label + "_ftd", dataset_file, epochs=200, batch_size=32)
    # # evaluate_model(fine_tunned, x_tr, y_test)
    # evaluate_model(fine_tunned, fine_tunned, x_tr, y_test)
