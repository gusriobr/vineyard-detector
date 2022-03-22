"""
Post processing to create feature file out of the raster images
"""
import logging
import os
from collections import OrderedDict

import fiona
import rasterio
from fiona.crs import from_epsg
from rasterio import features
from shapely.geometry import shape

import cfg

cfg.configLog()


def vectorize_predictions(raster_file, shape_file):
    logging.info("Started image vectorization from raster {} into shapefile {}".format(raster_file, shape_file))
    polys = None
    with rasterio.open(raster_file) as src:
        img = src.read()
        mask = img == 255
        polys = features.shapes(img, mask=mask, transform=src.transform)

    logging.info("Polygons extracted")

    # if file exist append otherwise create
    if os.path.exists(shape_file):
        open_f = fiona.open(shape_file, 'a')
    else:
        # output_driver = "GeoJSON"
        output_driver = 'ESRI Shapefile'
        vineyard_schema = {
            'geometry': 'Polygon',
            'properties': OrderedDict([('FID', 'int')])
        }
        crs = from_epsg(25830)
        open_f = fiona.open(shape_file, 'w', driver=output_driver, crs=crs, schema=vineyard_schema)

    with open_f as c:
        for p in polys:
            poly_feature = {"geometry": p[0], "properties": {"FID": 0}}
            c.write(poly_feature)

    logging.info("Vectorization finished.")


def filter_by_area(feature):
    area = shape(feature["geometry"]).area
    return area > 450


def filter_features(input_file, output_file):
    with fiona.open(input_file) as source:
        source_driver = source.driver
        source_crs = source.crs
        source_schema = source.schema
        polys_filtered = list(filter(filter_by_area, source))

    with fiona.open(output_file, "w", driver=source_driver, schema=source_schema, crs=source_crs) as dest:
        for r in polys_filtered:
            dest.write(r)


if __name__ == '__main__':
    input_folder = '/media/gus/data/viticola/raster/processed_v3'
    input_images = [os.path.join(input_folder, f_img) for f_img in os.listdir(input_folder) if f_img.endswith(".tif")]
    output_file = cfg.results("vineyard_polygons.shp")

    for f_image in input_images:
        vectorize_predictions(f_image, output_file)

    filtered_output_file = "/tmp/vineyard_filtered.shp"
    filter_features(output_file, filtered_output_file)

    logging.info("Filtered geometries successfully written");
