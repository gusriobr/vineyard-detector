import os
from pathlib import Path

import joblib
import pandas as pd
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator

import cfg
from vineyard.data import dataset

layers = tf.keras.layers

cfg.configLog()


def save(object, file_path):
    import jsonpickle
    json = jsonpickle.encode(object)
    # save json content
    fo = open(file_path, "w")
    fo.write(json)
    fo.close()


def train(model, dataset_file, output_folder, epochs=200, batch_size=32, min_delta=0.0001):
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    tf.keras.utils.plot_model(model, to_file=os.path.join(output_folder, 'model.png'), show_shapes=True,
                              show_layer_names=True, show_layer_activations=True, )

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
    save(model_conf, os.path.join(output_folder, "config.json"))

    model_path = os.path.join(output_folder, "model.md")
    metric_to_monitor = "val_accuracy"

    k_callbacks = [
        tf.keras.callbacks.ModelCheckpoint(model_path, monitor=metric_to_monitor, verbose=0, save_best_only=True,
                                           save_weights_only=False, mode='max', save_freq="epoch"),
        tf.keras.callbacks.ReduceLROnPlateau(monitor=metric_to_monitor, factor=0.1, patience=10, mode='auto',
                                             min_delta=min_delta, cooldown=0, min_lr=1e-6),
        tf.keras.callbacks.EarlyStopping(monitor=metric_to_monitor, patience=30, min_delta=min_delta)
    ]

    history = model.fit(
        train_gen,
        steps_per_epoch=len(y_train) // batch_size,
        epochs=epochs,
        validation_data=val_gen,
        validation_steps=len(y_test) // batch_size,
        callbacks=k_callbacks
    )
    # transform in panda dataframe and store as joblib file
    hist = dict(history.history)
    if 'lr' in hist:
        hist.pop("lr")  # remove key
    df_hist = pd.DataFrame(hist)

    history_file = os.path.join(output_folder, "history.jbl")
    joblib.dump(df_hist, history_file)


def fine_tunning(model_file):
    model = tf.keras.models.load_model(model_file)

    model.trainable = True
    model.compile(
        optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
    )
    return model
