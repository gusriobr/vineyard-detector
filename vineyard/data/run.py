import logging
import os
import shutil
from pathlib import Path

import cfg
from data.patches import extract_patches
from data.prepare_data import create_lirs
from vineyard.data import dataset

cfg.configLog()


def prepare_folder(folder):
    """
    Delete previous folders and create
    :param output_folder:
    :return:
    """
    folder_path = Path(folder)
    shutil.rmtree(folder, ignore_errors=True)
    # create folders
    folder_path.mkdir(parents=True, exist_ok=True)


if __name__ == '__main__':
    raster_folder = "/media/gus/data/viticola/raster"
    feature_file = cfg.resource('selectedParcels/selected_parcels.shp')
    dataset_folder = "/media/gus/data/viticola/datasets/dataset_v2"

    do_extract_lirs = False
    do_extract_patches = False
    do_create_dataset = True
    image_size = 48

    # extract feature lirs from raster filtering by feature file
    lirs_folder = os.path.join(dataset_folder, "lirs")
    Path(lirs_folder).mkdir(parents=True, exist_ok=True)
    if do_extract_lirs:
        logging.info("Extracting lirs from raster images")
        prepare_folder(lirs_folder)
        create_lirs(raster_folder, feature_file, lirs_folder)

    # extract patches from lirs
    patches_path = os.path.join(dataset_folder, "patches")
    Path(patches_path).mkdir(parents=True, exist_ok=True)
    if do_extract_patches:
        logging.info("Extracting patches from lirs")
        prepare_folder(patches_path)
        patch_options = {"size": image_size, "folder_per_category": True, "max_patches": 60}
        extract_patches(lirs_folder, patches_path, patch_options)

        # features with label 2 --> rename to mark as "no-vineyard" (folder 0)
        if os.path.exists(os.path.join(dataset_folder, "patches/2")):
            logging.info("Renaming patches output folder")
            os.rename(os.path.join(dataset_folder, "patches/2"), os.path.join(dataset_folder, "patches/0"))

    if do_create_dataset:
        # create numpy array with images
        logging.info("Creating dataset file: " + dataset_folder)
        dataset.create_dataset_file(patches_path, dataset_folder)

    logging.info("Dataset creation finished!")
