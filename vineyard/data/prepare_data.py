import logging
import os
import shutil
import tempfile
import uuid
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from matplotlib import pyplot as plt
from rasterio import features
from rasterio import mask
from shapely.geometry import Polygon

import cfg
from vineyard.data.patches import get_lir, extract_patches

cfg.configLog()

def rnd_name():
    return uuid.uuid4().hex[:10].upper().replace('0', 'X').replace('O', 'Y')


def locate_raster(raster_folder):
    file_list = os.listdir(raster_folder)
    return [os.path.join(raster_folder, name) for name in file_list]


def get_raster_bbox(rfile):
    """
    Get bounds from rasterio and craate rectangle polygon
    :param rfile:
    :return:
    """
    dataset = rasterio.open(rfile)
    bbox = dataset.bounds
    crs = dataset.crs
    dataset.close()
    # return poly
    return [bbox.left, bbox.bottom, bbox.right, bbox.top], "epsg:{}".format(crs.to_epsg())


def create_rectangles(raster_file, features, output_folder):
    """
    Receives a masked raster
    :param raster_file:
    :param features: list of tuples (id, category, geometry)
    :param output_folder:
    :return:
    """
    # burn geometries into image
    geometries = [f[2] for f in features]
    mask_output_folder = tempfile.gettempdir()
    masked_f = mask_features(raster_file, geometries, mask_output_folder)

    files = []

    with rasterio.open(masked_f) as src:
        for idx, feature in enumerate(features):
            fid, category, geometry = feature
            image, out_transform = mask.mask(src, [geometry], crop=True)
            # switch channels: bands  [0] --> image channels [2]
            image = np.moveaxis(image, 0, 2)

            image_rect = get_lir(image)
            if image_rect.shape[0] == 0:
                continue
            path = os.path.join(output_folder, "{}_{}.png".format(category, fid))
            files.append(path)
            plt.imsave(path, image_rect)
    return files


def burn_features(raster_file, geometries, output_folder):
    # get raster metadata from source
    source = rasterio.open(raster_file)

    meta = source.meta.copy()
    meta.update(compress='lzw', count=1)  # just one band as output
    source.close()

    output_f = os.path.join(output_folder, "burned.tif")
    # burn features on the raster and create a mask image
    with rasterio.open(output_f, 'w+', **meta) as out:
        shapes = ((geom, 255) for geom in geometries)
        burned = features.rasterize(shapes=shapes, fill=0, out_shape=out.shape, transform=out.transform)
        out.write_band(1, burned)
    return output_f


def mask_features(raster_file, geometries, output_folder):
    """
    Takes the input raster and maskes the background of the given geometries to create a new raster named "masked.tif"
    :param raster_file: input file
    :param geometries: geometries to mask
    :param output_folder:  output raster destination
    :return:
    """
    output_f = os.path.join(output_folder, "masked.tif")
    with rasterio.open(raster_file) as src:
        # out_image, out_transform = rasterio.mask.mask(src, geometries, crop=True)
        out_image, out_transform = rasterio.mask.mask(src, geometries)
        out_meta = src.meta

    out_meta.update({"driver": "GTiff"})  # , "transform": out_transform})
    with rasterio.open(output_f, "w", **out_meta) as dest:
        dest.write(out_image)

    return output_f


def read_features(feature_file, bbox, feature_options=None):
    """
    Read feature file filtering geometries by the given bbox
    :param feature_file:
    :param bbox:
    :param feature_options:
    :return: list of tuples with (id, category, geometry)
    """

    gdf = gpd.read_file(feature_file, bbox=bbox)
    # bbox returns geometries that intersect with the bbox if filter_bype=contains, filter the returned geometries to
    # get just the ones strictly contained in the bbox
    if feature_options and feature_options.get("filter_type", "bbox") == "contains":
        bbox_poly = Polygon.from_bounds(*bbox)
        gdf = gdf[gdf.geometry.within(bbox_poly)]

    # get geometries as list
    id_attribute = feature_options.get("id_attribute", "ogc_fid") if feature_options else "ogc_fid"
    geo_attribute = feature_options.get("geo_attribute", "geometry") if feature_options else "geometry"
    category_attribute = feature_options.get("category_attribute", "category") if feature_options else "category"
    return list(
        zip(gdf[id_attribute].tolist(), gdf[category_attribute].tolist(), gdf[geo_attribute].tolist())), gdf.crs.srs


def create_lirs(raster_folder, feature_file, lirs_folder,
                feature_options={"geo_attribute": "geometry", "category_attribute": "category",
                                    "filter_bype": "bbox"}):
    """
    Extracts patches from a raster using the feature file as filter.
    For each raster file contained in the raster folder, the containing geometries are loaded to mask the image,
    the, for each image, the Largest Rectangle Inside (lir) is created and stored in folder output_folder/lirs.
    These lirs will be used to extract the patches using sliding window technique and stored in output_folder/patches.
    :param feature_options:
    :param raster_folder:
    :param feature_file:
    :return:
    """
    if not os.path.exists(feature_file):
        raise FileNotFoundError("Feature file doesn't exists: " + feature_file)

    raster_files = locate_raster(raster_folder)

    logging.info('Found {} raster files'.format(len(raster_files)))

    for rfile in raster_files:
        logging.info('Processing raster file: ' + rfile)
        # get metadata from the file to filter features and get the
        bbox, raster_srs = get_raster_bbox(rfile)
        feature_list, geom_srs = read_features(feature_file, bbox, feature_options)
        if len(feature_list) == 0:
            continue
        if raster_srs != geom_srs:
            raise ValueError(
                "Invalid CRS, raster and feature files must be in the same CRS: "
                "raster={} features={}".format(raster_srs, geom_srs))

        create_rectangles(rfile, feature_list, lirs_folder)
