import logging
import os
import random
import shutil
from pathlib import Path

import cfg

try:
    import _pickle as pickle  # cPickle
except:
    import pickle

import numpy as np
from keras.preprocessing.image import img_to_array, load_img

cfg.configLog()

def split_patches(base_folder, output_folder, split=0.3, shuffle=True):
    """
    Splits the patches folder into train and test folders. Takes the original patches folder, creates a "splits" folder
    with two nested directories "train/" and "test/", each of them having the corresponding class folders
    (ej: label1/, label2/).
    Output_folder will be deleted during the process.
    :param base_folder: root of the classes folders. There must be a folder per class ej label1/, label2/
    :return:
    """
    # delete splits folder if already exists
    splits_path = Path(output_folder)
    shutil.rmtree(splits_path, ignore_errors=True)
    splits_path.mkdir(parents=True, exist_ok=True)
    train_folder = os.path.join(output_folder, "train")
    test_folder = os.path.join(output_folder, "test")
    Path(train_folder).mkdir(parents=True, exist_ok=True)
    Path(test_folder).mkdir(parents=True, exist_ok=True)

    # iterate over categories' folders coping the files
    labels = os.listdir(base_folder)
    for label in labels:
        os.makedirs(os.path.join(output_folder, "train", label))
        os.makedirs(os.path.join(output_folder, "test", label))

        category_folder = os.path.join(base_folder, label)
        # get files from category filder
        files = [f for f in os.listdir(category_folder) if os.path.isfile(os.path.join(category_folder, f))]
        max_train_images = int(len(files) * (1 - split))

        if shuffle:
            random.shuffle(files)
        dst_folder = os.path.join(output_folder, "train", label)
        for idx, img_file in enumerate(files):
            shutil.copyfile(os.path.join(category_folder, img_file), os.path.join(dst_folder, img_file))
            if idx == max_train_images - 1:
                dst_folder = os.path.join(output_folder, "test", label)


def load_images_as_array(folder: object):
    images = [os.path.join(folder, f) for f in os.listdir(folder)]

    arr = None
    for idx, img_file in enumerate(images):
        img = load_img(img_file)
        # convert to numpy array
        img_array = img_to_array(img)
        if arr is None:
            arr = np.zeros((len(images), img_array.shape[0], img_array.shape[1], 3), np.uint8)
        arr[idx, :] = img_array
    return arr


def load_batch(splits_folder):
    """
    Iterates over subdirectories reading all images to create two arrays, one with the image data (X data)
    and other with the labels (folder names must be integers) as y array.
    :param splits_folder:
    :return:
    """
    #
    labels = [f for f in os.listdir(splits_folder) if not os.path.isfile(os.path.join(splits_folder, f))]
    full_arr = None
    for label in labels:
        logging.info(">> processing label: " + label)
        arr = load_images_as_array(os.path.join(splits_folder, label))
        labels_arr = np.zeros((arr.shape[0], 1), np.uint8)
        labels_arr[:, 0] = int(label)
        if full_arr is None:
            full_arr = arr
            full_label = labels_arr
        else:
            full_arr = np.vstack([full_arr, arr])
            full_label = np.vstack([full_label, labels_arr])
        # create labels array
    # shuffle
    perm = np.random.permutation(full_arr.shape[0])
    return full_arr[perm, :], full_label[perm, :]


def create_dataset_file(patches_path, output_folder, split=0.3):
    """
    Takes patches folders, split the images into train/ and test/ folders and stores the images
    into a numpy array structure with the typical data configuration:
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    where x_train, x_test are uint8 [n_images, size, size, 3)
    and y_train, y_test are  uint8 [n_images, 1)
    :param patches_path:
    :return:
    """
    splits_folder = os.path.join(output_folder, "splits")
    split_patches(patches_path, splits_folder, split=split)
    # train
    logging.info("Loading training batch")
    x_train, y_train = load_batch(os.path.join(splits_folder, "train"))
    logging.info("Loading test batch")
    x_test, y_test = load_batch(os.path.join(splits_folder, "test"))

    out_file = os.path.join(output_folder, "dataset.npy")
    logging.info("Creating data file " + out_file)
    with open(out_file, 'wb') as handle:
        pickle.dump(((x_train, y_train), (x_test, y_test)), handle)


def load(dataset_file):
    with open(dataset_file, 'rb') as handle:
        data = pickle.load(handle)
    return data
