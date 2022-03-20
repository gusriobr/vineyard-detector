import logging
import os
from pathlib import Path

import tensorflow as tf

tf.config.set_visible_devices([], 'GPU')

import cfg
from data.model import DatasetDef
from data.raster import standarize_dataset
from evaluation.activations import visualize_filters
from evaluation.eval import evaluate_model_file
from models.cnn import BasicCNN
from train.run_train import train, img_augmentation
from vineyard.data import dataset

if __name__ == '__main__':
    ############################
    ### DESACTIVATE GPU ####
    ############################
    # Hide GPU from visible devices
    ############################

    logging.info("Starting training process")

    dataset_file = "/media/gus/data/viticola/datasets/dataset_v3/dataset.npy"
    base_output = cfg.results("iteration2")

    logging.info("Using dataset " + dataset_file)

    train_model = False
    eval_model = False
    tune_model = False
    eval_tune = False
    plot_filters = True

    model = None
    model_path = None

    train_epochs = 10
    IMG_SIZE = 64

    model_label = "cnnv1"
    model_folder = os.path.join(base_output, model_label)
    Path(model_folder).mkdir(parents=True, exist_ok=True)
    if train_model:
        logging.info("Training model: " + model_label)
        # model = build_binary_classifier(augmentation_layer=img_augmentation, freeze_weights=True, image_size=IMG_SIZE)
        model = BasicCNN.buildv1(IMG_SIZE, IMG_SIZE, 3, 1, augmentation_layer=img_augmentation)

        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
        train(model, dataset_file, model_folder, epochs=train_epochs, batch_size=64)

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
        train(model, dataset_file, model_folder, epochs=train_epochs, batch_size=96)

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

    logging.info("Training process finished successfully!")
