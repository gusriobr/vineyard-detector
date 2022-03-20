# -*- coding: utf-8 -*-

import math
import os
from collections import namedtuple
from pathlib import Path

import numpy as np
import pyproj
from shapely import geometry
from shapely.ops import transform
from sklearn.feature_extraction.image import extract_patches_2d

from vineyard.utils import lir
from skimage.io import imread, imsave


def extract_patches(input_folder, output_folder, patch_options=None):
    """
    Given an imagen folder, where each image name have as prefix its category name, creates patches for the imagen,
    creates patches for the imagen keeping as prefix the image file name.
    :param input_folder: folder containing input images
    :param output_folder: base folder to store images
    :param patch_options:
    :param folder_per_category: if a directory has to be created to keep each category patches apart
    :param size: size of the patches
    :param step_size: gap between patches when sliding window
    :param patches_folder:
    :
    :return:
    """
    # create a folder per category?
    if patch_options is None:
        patch_options = {}
    folder_per_category = patch_options.get("folder_per_category", False)
    size = patch_options.get("size", 64)
    max_patches = patch_options.get("max_patches", 50)

    files = os.listdir(input_folder)
    if len(files) == 0:
        raise Exception("The LIRs folder is empty!")

    for img in os.listdir(input_folder):
        img_f = os.path.join(input_folder, img)
        category, f_name = get_prefix(img_f)

        output_f = output_folder if not folder_per_category else os.path.join(output_folder, category)
        Path(output_f).mkdir(parents=True, exist_ok=True)  # create if not exists

        img = imread(img_f)
        if min(img.shape[:2]) < size:
            continue  # image is too small
        patches = extract_patches_2d(img, (size, size), max_patches=max_patches)
        # patches = list(sliding_window(img, step_size=step_size, window_size=(size, size)))
        # store patches in folder
        for idx, p_img in enumerate(patches):
            path_name = os.path.join(output_f, f_name.replace(".", "_{}.".format(idx)))
            print(path_name)
            imsave(path_name, p_img)


def sliding_window(image, step_size=5, window_size=(32, 32)):
    """
    slide a window across the image extracting the images
    :param image: 
    :param step_size: 
    :param window_size: 
    :return: 
    """
    for x in range(0, image.shape[1] - window_size[0], step_size):
        for y in range(0, image.shape[0] - window_size[1], step_size):
            yield image[y:y + window_size[1], x:x + window_size[0], :]


def get_prefix(file_name):
    splits = os.path.basename(file_name).split("_")
    # category_name.extension
    return splits[0], "_".join(splits[1:])


def get_lir(image):
    """
    Gets
    Largest
    Inside
    Rectangle
    for the given geometry
    :param
    image: reference
    image
    :return:
    """
    # turn image into 0-1 matrix
    im_mask = np.dot(image[..., :3], [1, 1, 1])
    im_mask[im_mask > 0] = 1
    im_mask = im_mask.astype(np.uint8)

    rect = lir.largest_interior_rectangle(im_mask)
    image_sqr = image[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2], :]
    return image_sqr


def project_geo(geo, s_crs, d_crs):
    origin_crs = pyproj.CRS(s_crs)
    dest_crs = pyproj.CRS(d_crs)

    project = pyproj.Transformer.from_crs(origin_crs, dest_crs, always_xy=True).transform
    return transform(project, geo)


def getCellMatrix(geom, size):
    minx, miny, maxx, maxy = geom.bounds
    width = round(maxx - minx)
    height = round(maxy - miny)
    nWidth = int(math.floor(width / size))
    nHeight = int(math.floor(height / size))

    grid = np.zeros((nHeight, nWidth), np.uint8)

    for i in range(height):
        for j in range(width):
            # Creamos la celda actual
            current_minx = minx + j * size
            current_maxx = minx + (j + 1) * size
            current_maxy = maxy - i * size
            current_miny = maxy - (i + 1) * size

            # Creamos el polígono de la celda
            cell = geometry.box(current_minx, current_miny, current_maxx, current_maxy)

            # Comprobamos si la celda está completamente contenida en el polígono
            if geom.contains(cell):
                grid[i, j] = 1
    return grid


if __name__ == '__main__':
    from shapely.geometry import shape
    from matplotlib import pyplot as plt

    geo = {"type": "Polygon", "coordinates": [
        [[-4.802742004394531, 41.56954541260106], [-4.79527473449707, 41.5654034571327],
         [-4.773473739624023, 41.58168077630336], [-4.780383110046386, 41.58723403139947],
         [-4.802742004394531, 41.56954541260106]]]}

    poly = shape(geo)
    poly = project_geo(poly, "EPSG:4258", "EPSG:25830")
    cells = getCellMatrix(poly, 10)

    rect = lir.largest_interior_rectangle(cells)
    # box = x, y, width, height
    box = cells[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]
    plt.imshow(box)
    plt.show()
