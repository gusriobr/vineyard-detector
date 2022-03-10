# USAGE
# python build_dataset.py

import logging
import os
import random
import shutil

import numpy as np
from PIL import Image as pil_image
from imutils import paths
# import the necessary packages
from keras.preprocessing.image import save_img, img_to_array
from scipy import misc

# from srs.data.py.utils import list_images_in_folders
from vineyard.data.model import DatasetDef

logger = logging.getLogger(__name__)


def list_images_in_folders(folders, max_img_folder=None, shuffle_files=True):
    imagePaths = []
    for folder in folders:
        lst = list(paths.list_images(folder))
        if shuffle_files:
            random.shuffle(lst)
        if max_img_folder is not None:
            lst = lst[:min(len(lst), max_img_folder)]
        imagePaths += lst
    return imagePaths


def clean_folders(dts_def):
    """
    Re-creates the folder structure
    :return:
    """
    logger.info("[INFO] cleaning folders...")
    base = dts_def.base_folder
    shutil.rmtree(dts_def.get_train_folder(), ignore_errors=True)
    shutil.rmtree(dts_def.get_test_folder(), ignore_errors=True)

    logger.info("[INFO] creating folder structure.")
    if not os.path.exists(dts_def.base_folder):
        os.mkdir(base)

    main_folders = ["train", "test"]
    # ex: lr_96, lr_48, lr_32, lr_24

    if dts_def.patch_is_gt:
        scale_folders = ["lr_{}".format(int(dts_def.patch_size / scale)) for scale in dts_def.scales]
    else:
        # the reference is the lowest resolution image
        scale_folders = ["lr_{}".format(int(dts_def.patch_size * scale)) for scale in dts_def.scales]

    for mf in main_folders:
        # create train/test folders
        if not os.path.exists(os.path.join(base, mf)):
            os.mkdir(os.path.join(base, mf))
        for scale in scale_folders:
            os.mkdir(os.path.join(base, mf, scale))


def extract_images(imagePaths, dts_def, max_images=None, intermediate_folder=None,
                   interpolation="bicubic", max_patches_image=None):  # hr_dim=96, scales=[1, 2, 3, 4]):
    """
    Extract image patches from the raster files passed in the "imagePaths" argument
    :param imagePaths: list of raster file names
    :param dts_def: data.py definition object
    :param max_images: max number of images to extract for the data.py
    :param intermediate_folder: temporary folder for debugging purposes
    :param interpolation: type to interpolation used to generate pyramids
    :param max_patches_image: max patches per raster
    :return:
    """

    scales = dts_def.scales

    if not dts_def.patch_is_gt:
        max_scale = max(scales)
        scales = [sc / max_scale for sc in scales]

    # # ground-truth path size
    # hr_base = dts_def.patch_size * dts_def.gt_downscale

    idx = 0
    total = 0
    if max_images is not None:
        print("Max num images to generate: {}".format(max_images))
    num_images = len(imagePaths)
    for nimg, imagePath in enumerate(imagePaths):
        print(">>> Processing image {} of {}:  {}".format(nimg + 1, num_images, imagePath))
        # load the input image
        # x = load_img(imagePath)
        x = pil_image.open(imagePath)
        image = img_to_array(x, data_format="channels_last")
        # image = cv2.imread(imagePath)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # grab the dimensions of the input image and crop the image such
        # that it tiles nicely when we generate the training data.py +
        # labels

        # calculate max patch size
        if dts_def.patch_is_gt:
            # max size is the patch size
            hr_dim = dts_def.patch_size
        else:
            # max size is calculated using the lowest resolution image size
            hr_dim = dts_def.patch_size * max(dts_def.scales)

        if dts_def.gt_downscale > 1:
            # downsample the base image
            # image = cv2.resize(image, (hr_dim,) * 2, interpolation=cv2.INTER_CUBIC)
            image = misc.imresize(image, 1.0 / dts_def.gt_downscale, interp=interpolation)

        (h, w) = image.shape[:2]
        w -= int(w % hr_dim)
        h -= int(h % hr_dim)

        image = image[0:h, 0:w]

        HR_W = w  # hr_dim * 2
        HR_H = h  # hr_dim * 2

        # to generate our training images we first need to downscale the
        # image by the scale factor...and then upscale it back to the
        # original size -- this will process allows us to generate low
        # resolution inputs that we'll then learn to reconstruct the high
        # resolution versions from

        # slide a window from left-to-right and top-to-bottom

        # if the base image has a greater gsd than the desired gt,
        # we have to scale it before we use it as gt

        for scale in scales:

            if dts_def.patch_is_gt:
                LR_DIM = int(hr_dim / scale)
                h = int(HR_H / scale)
                w = int(HR_W / scale)
                if scale == min(scales):
                    scaled = image
                else:
                    scaled = misc.imresize(image, 1.0 / scale, interp=interpolation)
            else:
                LR_DIM = int(hr_dim * scale)
                h = int(HR_H * scale)
                w = int(HR_W * scale)
                if scale == max(scales):
                    scaled = image
                else:
                    scaled = misc.imresize(image, scale, interp=interpolation)

            # store scaled image to check
            if intermediate_folder:
                name = os.path.basename(imagePath)
                name, ext = os.path.splitext(name)
                lr_image_path = os.path.join(intermediate_folder, "{}__{}{}".format(name, scale, ext))
                save_img(lr_image_path, scaled)

            STRIDE = int(LR_DIM * dts_def.stride)
            # restart idx on each scale
            idx = total
            num_patches_image = 0
            break_exit = False
            for y in range(0, h - LR_DIM + 1, STRIDE):
                for x in range(0, w - LR_DIM + 1, STRIDE):
                    if idx == max_images:
                        break_exit = True
                        break
                    if max_patches_image is not None \
                            and num_patches_image > max_patches_image:
                        break_exit = True
                        break
                    # crop output the `INPUT_DIM x INPUT_DIM` ROI from our
                    # scaled image -- this ROI will serve as the input to our
                    # network
                    crop = scaled[y:y + LR_DIM, x:x + LR_DIM]
                    lr_output_folder = "lr_{}".format(LR_DIM)

                    # construct the crop and target output image paths
                    crop_path = os.path.join(dts_def.get_train_folder(), lr_output_folder, "{}.png".format(idx))

                    # write the images to disk
                    save_img(crop_path, crop)

                    # increment the image index
                    idx += 1
                    num_patches_image += 1
                if break_exit:
                    break;
            print("Scale / Counters: {}/{} ".format(scale, idx))
        total = idx
        if total == max_images:
            break;
    dts_def.n_train_images = total


def extract_random_test_set(dts_def, test_split=0.3):
    """
    Extract part of the data.py to the test folders
    :return:
    """
    logger.info("[INFO] extracting test set...")

    num_images = dts_def.get_total_images()
    # get part of random numbers, don't care about duplicates
    num_img_selected = int(num_images * test_split)
    selected = np.random.random_integers(0, num_images - 1, size=num_img_selected)
    selected = set(selected)
    for i in selected:
        for scale in dts_def.scales:

            if dts_def.patch_is_gt:
                scale_folder = int(dts_def.patch_size / scale)
            else:
                scale_folder = int(dts_def.patch_size * scale)
            img_path = os.path.join(dts_def.get_train_folder(),
                                    "lr_{}/{}.png".format(scale_folder, i))

            # move image from train to test folder
            os.rename(img_path, img_path.replace("train", "test"))

    dts_def.n_test_images = len(selected)
    dts_def.n_train_images -= dts_def.n_test_images


if __name__ == '__main__':
    # plot_histogram('/home/gus/workspaces/datasets/srs/dts_test/train/lr_120/1.png', '120')
    # exit(0)
    # base_folder = '/media/data.py/rasters/aerial/pnoa/2011/color' # evaluation
    # base_folder = '/media/data.py/rasters/aerial/pnoa/2010/color'
    base_folder = '/data.py/raster/pnoa/2010/color'

    input_folders = [
        # validation
        # base_folder + "/H-0371",
        base_folder + "/H-0372",
        base_folder + "/H-0373",
        base_folder + "/H-0400",
        # home
        # base_folder + "/H-0401",
        # base_folder + "/H-0457",
        # base_folder + "/H-0483",
        # itacyl
        base_folder + "/H-0431",
        base_folder + "/H-0556",

        # ir combined images
        # '/media/apps/rasters/pnoa/2010/combined'
        # topillo
        # "/home/gus/workspaces/datasets/srs/input_hr"
    ]

    dts_def = DatasetDef(120, '/data.py/data.py/pnoa_200K',  # gt_downscale=10,
                         patch_is_gt=True, scales=[1, 2, 4], max_img_folder=100, stride=4)

    print("Building data.py destination folder: {}".format(dts_def.base_folder))
    clean_folders(dts_def)
    imagePaths = list_images_in_folders(input_folders, dts_def.max_img_folder, dts_def.max_img_folder)
    # imagePaths = imagePaths[:1]
    # imagePaths = ['/media/apps/rasters/pnoa/2010/combined/0372_1-2.tif']
    print("Number of input images: {}".format(len(imagePaths)))

    # imagePaths = imagePaths[:1]
    np.random.shuffle(imagePaths)

    MAX_NUM_IMAGES = 200000  # 2M
    TEST_SPLIT = 0.25
    extract_images(imagePaths, dts_def, max_images=MAX_NUM_IMAGES,
                   max_patches_image=300)  # scales=scales, hr_dim=hr_dim)
    print("Selecting {}% test images".format(TEST_SPLIT * 100))
    extract_random_test_set(dts_def, test_split=TEST_SPLIT)
    print("Num images Train: {}, Test: {}, Total: {}".format(dts_def.n_train_images,
                                                             dts_def.n_test_images,
                                                             dts_def.get_total_images()))

    # build_hd5(dts_def)
    # build_hd5(config.TEST_FOLDER)

    logger.info("[INFO] finished.")

    dts_def.save()
