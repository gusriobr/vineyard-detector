import logging
import os
from pathlib import Path

import tensorflow as tf

import cfg
from data.model import DatasetDef
from data.raster import standarize_dataset
from evaluation.activations import visualize_filters
from evaluation.eval import evaluate_model_file, summarize
from models.cnn import BasicCNN
from models.transferlr import build_model_f
from train import train
from vineyard.data import dataset

layers = tf.keras.layers

cfg.configLog()

kwargs = {}

img_augmentation = tf.keras.models.Sequential([
    layers.RandomContrast(factor=0.2),
    layers.RandomZoom(height_factor=(-0.2, 0.2))
], name="img_augmentation")


def build_cc():
    return BasicCNN.build(IMG_SIZE, IMG_SIZE, 3, num_classes=2, augmentation_layer=img_augmentation)


if __name__ == '__main__':
    dataset_file = "/media/gus/data/viticola/datasets/dataset_v2/dataset.npy"
    base_output = cfg.results("iteration4")

    train_model = True
    eval_model = False
    tune_model = False
    eval_tune = False
    plot_filters = False

    model = None
    model_path = None

    train_epochs = 200
    IMG_SIZE = 48
    batch_size = 64

    logging.info("Starting training process")

    model_defs = [
        # build_model_f("InceptionV3", img_augmentation, IMG_SIZE),
        ["cnnv1", build_cc],
        ["vgg19", build_model_f("vgg19", img_augmentation, IMG_SIZE)],
        # ["Xception", build_model_f("Xception", img_augmentation, IMG_SIZE)],
        # ["effNet_64", build_model_f("EfficientNetB0", img_augmentation, IMG_SIZE)],
        # ["InceptionV3", build_model_f("InceptionV3", img_augmentation, IMG_SIZE)],
        # ["ResNet50", build_model_f("ResNet50", img_augmentation, IMG_SIZE)],
    ]

    for model_def in model_defs:
        tf.keras.backend.clear_session()

        model_label = model_def[0]
        model_builder = model_def[1]
        model_folder = os.path.join(base_output, model_label)
        Path(model_folder).mkdir(parents=True, exist_ok=True)
        try:
            if train_model:
                logging.info(">>> Training model: " + model_label)
                model = model_builder()
                optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
                model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
                train(model, dataset_file, model_folder, epochs=train_epochs, batch_size=batch_size)
                logging.info("Training finished for: " + model_label)

            model_path = os.path.join(model_folder, "model.md")
            if eval_model:
                # standarize test images images
                logging.info("Evaluating model: " + model_label)
                (x_train, y_train), (x_test, y_test) = dataset.load(dataset_file)

                model_conf_file = os.path.join(model_folder, "config.json")
                model_conf = DatasetDef.read(model_conf_file)
                x_tr = standarize_dataset(x_test, model_conf["mean"], model_conf["std"])
                evaluate_model_file(model_path, x_tr, y_test)

            if plot_filters:
                logging.info("Creating filter visualization for model: " + model_label)
                output_vis = os.path.join(model_folder, "vis")
                if model is None:
                    model = tf.keras.models.load_model(model_path)
                visualize_filters(model, output_vis, image_size=IMG_SIZE)

            if tune_model:
                if model is None:
                    model = tf.keras.models.load_model(model_path)

                logging.info("Fine tuning model " + model_path)
                model.trainable = True
                model_label = model_label + "_ftd"
                model_folder = os.path.join(base_output, model_label)
                train(model, dataset_file, model_folder, epochs=train_epochs, batch_size=batch_size)

                if plot_filters:
                    output_vis = os.path.join(model_folder, "vis")
                    visualize_filters(model, output_vis, image_size=IMG_SIZE)

                if eval_tune:
                    # standarize test images images
                    if x_test is None:
                        (x_train, y_train), (x_test, y_test) = dataset.load(dataset_file)

                    model_conf_file = os.path.join(model_folder, "config.json")
                    model_conf = DatasetDef.read(model_conf_file)
                    x_tr = standarize_dataset(x_test, model_conf["mean"], model_conf["std"])
                    evaluate_model_file(model_path, x_tr, y_test)
        except Exception as e:
            logging.exception("Error while processing model " + model_label)
            logging.critical(e, exc_info=True)

        logging.info("Training process finished successfully!")

        summarize(base_output)

        logging.info("Model stats summarized!")
