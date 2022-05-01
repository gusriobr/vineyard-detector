import glob
import logging
import os
import sys

import cv2
import numpy as np
import skimage.io
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
from skimage import io
from tensorflow.python.keras import backend as K

ROOT_DIR = os.path.abspath("../../")
sys.path.append(ROOT_DIR)

from vineyard import cfg
from vineyard.data.model import DatasetDef
from vineyard.data.raster import georeference_image
from vineyard.models.cnn import BasicCNN

layers = tf.keras.layers


cfg.configLog()


def comp_file(filename):
    return os.path.join(cfg.root_folder(), filename)


def get_file(filepath):
    return cfg.file_path(filepath)
    # basepath = os.path.dirname(os.path.abspath(__file__))
    # return os.path.join(basepath, filepath)


def get_folder(filepath):
    f_path = cfg.file_path(filepath)
    if not os.path.exists(f_path):
        os.makedirs(f_path)
    return f_path


def pred_to_category(value, threshold=0.80):
    v = 255 if value >= threshold else 0
    return np.array([v], dtype=np.uint8)


def standarize(mean, std):
    def f_inner(x):
        x_tr = x * (1.0 / 255.0)
        x_tr -= mean
        x_tr /= std
        return x_tr

    return f_inner


def predict(model, x, datagen, std_f):
    if std_f is not None:
        x = std_f(x)
        return model.predict(x)
    else:
        predict_gen = datagen.flow(x, y=None, shuffle=False)
        return model.predict_generator(predict_gen)


def extract_images(image_path, patch_size, output_path, model, datagen=None, channels=[0, 1, 2],
                   scale=1, treat_borders=True, std_f=None):  # hr_dim=96, scales=[1, 2, 3, 4]):
    """
        Iterate over the image to extract patches
    """
    nchannels = len(channels)

    # # ground-truth path size
    # hr_base = dts_def.patch_size * dts_def.gt_downscale

    # load the input image
    image = read_img(image_path)

    # select channels
    image = image[:, :, channels]

    # grab the dimensions of the input image and crop the image such
    # that it tiles nicely when we generate the training data +
    # labels
    (h, w) = image.shape[:2]
    if treat_borders:
        rest_w = int(w % patch_size)
        rest_h = int(w % patch_size)
    else:
        rest_w = 0
        rest_h = 0
        w -= int(w % patch_size)
        h -= int(h % patch_size)
        image = image[0:h, 0:w]

    output_img = np.zeros((h * scale, w * scale, 1), dtype=np.uint8)

    output_patch = patch_size * scale
    batch_size = w // patch_size
    for y in range(0, h - patch_size + 1, patch_size):
        batch = np.zeros((batch_size, patch_size, patch_size, nchannels))
        i = 0
        for x in range(0, w - patch_size + 1, patch_size):
            batch[i] = image[y:y + patch_size, x:x + patch_size]
            i += 1

        # apply model
        y_pred = predict(model, batch, datagen, std_f)

        # paste in the original image
        i = 0
        y_output = y * scale
        for x in range(0, w * scale - output_patch + 1, output_patch):
            output_img[y_output:y_output + output_patch, x:x + output_patch] = pred_to_category(y_pred[i])
            i += 1

    if rest_h > 0 and treat_borders:
        # srs_row(model, image, output_img, patch_size, output_patch, batch_size, nchannels, scale, w, y)
        batch = np.zeros((batch_size, patch_size, patch_size, nchannels), dtype=np.uint8)
        i = 0
        y = image.shape[0] - patch_size
        for x in range(0, w - patch_size + 1, patch_size):
            batch[i] = image[y:y + patch_size, x:x + patch_size]
            i += 1
        # apply model
        y_pred = predict(model, batch, datagen, std_f)
        # paste in the original image
        i = 0
        y_output = y * scale
        for x in range(0, w * scale - output_patch + 1, output_patch):
            output_img[y_output:y_output + output_patch, x:x + output_patch] = pred_to_category(y_pred[i])
            i += 1

    if rest_w > 0 and treat_borders:
        batch_size = h // patch_size
        batch = np.zeros((batch_size, patch_size, patch_size, nchannels))
        i = 0
        x = image.shape[1] - patch_size
        for y in range(0, h - patch_size + 1, patch_size):
            batch[i] = image[y:y + patch_size, x:x + patch_size]
            i += 1

        # apply model
        y_pred = predict(model, batch, datagen, std_f)

        # paste in the original image
        i = 0
        x_output = x * scale
        for y in range(0, h * scale - output_patch + 1, output_patch):
            output_img[y:y + output_patch, x_output:x_output + output_patch] = pred_to_category(y_pred[i])
            i += 1

    if rest_h > 0 or rest_w > 0 and treat_borders:
        # bottom right corner
        batch = np.zeros((1, patch_size, patch_size, nchannels))
        y = image.shape[0] - patch_size
        x = image.shape[1] - patch_size
        batch[0] = image[y:y + patch_size, x:x + patch_size]

        y_pred = predict(model, batch, datagen, std_f)

        x_output = x * scale
        y_output = y * scale
        output_img[y_output:y_output + output_patch, x_output:x_output + output_patch] = pred_to_category(y_pred[0])

    skimage.io.imsave(output_path, output_img)


def read_img(path):
    rimg = io.imread(path)
    return rimg


def build_model(model_folder, img_size=48):
    """
    :param model_folder:
    :param img_size:
    :return:
    """
    checkpoint_path = os.path.join(model_folder, "model.md")
    img_augmentation = tf.keras.models.Sequential([
        layers.RandomContrast(factor=0.1),
        layers.RandomZoom(height_factor=(-0.2, 0.2))
    ], name="img_augmentation")

    loaded_model = BasicCNN.build(img_size, img_size, 3, num_classes=2, augmentation_layer=img_augmentation)
    loaded_model.load_weights(checkpoint_path)

    return loaded_model


def load_pnoa_filenames(base_folder, tile_file):
    """
    Localiza tiles del pnoa a partir de fichero
    :param base_folder:
    :return:
    """
    lines = open(tile_file).read().splitlines()
    files = set()
    for line in lines:
        if line:  # not empty
            # is a file
            fabs = "{}/{}"
            if os.path.exists(fabs) and os.path.isfile(fabs):
                files.add(fabs)
            else:
                nested_files = glob.glob("{}/{}/*.tif".format(base_folder, line))
                if len(nested_files) > 0:
                    # it has nested tif
                    files.update(nested_files)
                else:
                    # it has nested folders with tifs
                    nested_files = glob.glob("{}/{}/**/*.tif".format(base_folder, line))
                    files.update(nested_files)
    lst_files = list(files)
    lst_files.sort()
    return lst_files


if __name__ == '__main__':
    # load srs model
    models = [
        ['/workspaces/wml/vineyard-detector/results/iteration4/cnnv1/', 'cnnv1', 1],
    ]
    # input_folder = '/media/gus/data/rasters/aerial/pnoa/2020/'
    # output_folder = '/media/gus/data/viticola/raster/processed_v4'
    input_folder = '/media/cartografia/01_Ortofotografia/'
    output_folder = '/workspaces/wml/vineyard-detector/results/processed_v4/2021'

    # find all nested images
    input_images = load_pnoa_filenames(input_folder, cfg.project_file('vineyard/inference/pnoa_files.txt'))
    input_images.sort()

    # input_images = ['/workspaces/wml/vineyard-detector/resources/PNOA_CYL_2020_25cm_OF_etrsc_rgb_hu30_h05_0345_4-6.tif']

    patch_size = 48
    for m in models:
        # clear tensorflow memory
        K.clear_session()
        tf.keras.backend.clear_session()
        model_path = m[0]
        tag = m[1]  # model_name

        model = build_model(model_path)

        datagen = ImageDataGenerator(rescale=1. / 255, featurewise_center=True, featurewise_std_normalization=True)
        model_conf = DatasetDef.read(os.path.join(model_path, "config.json"))
        datagen.mean = model_conf["mean"]
        datagen.std = model_conf["std"]
        std_func = standarize(model_conf["mean"], model_conf["std"])

        total = len(input_images)
        for idx, input in enumerate(input_images):
            logging.info("Processing image {} of {} - {}".format(idx, total, input))
            filename = os.path.basename(input)
            base, ext = os.path.splitext(filename)
            outf = os.path.join(output_folder, "{}_{}{}".format(base, tag, ext))

            extract_images(input, patch_size, outf, model, std_f=std_func, channels=[0, 1, 2], scale=1)

            logging.info("Applying geolocation info.")
            rimg = read_img(outf)
            rimg = rimg[:, :, np.newaxis]
            georeference_image(rimg, input, outf, scale=1, bands=1)
            logging.info("Finished processing file {}, \ngenerated output raster {}.".format(input, outf))

    # plt.show()
    logging.info("========================================")
    logging.info("Model inference  on raster finished.")
    logging.info("========================================")
