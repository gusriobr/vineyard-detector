import os.path
import tempfile
import unittest

import geopandas as gpd
import rasterio

import test.cfg_test as tcfg
from vineyard.data.prepare_data import get_raster_bbox, read_features, mask_features, create_rectangles, \
    burn_features


class TestPrepareData(unittest.TestCase):

    def test_get_raster_bbox(self):
        raster_file = tcfg.resource("aerial1.tif")
        bbox, srs = get_raster_bbox(raster_file)
        self.assertTrue(isinstance(bbox, list))
        self.assertTrue(4, len(bbox))
        self.assertEqual("epsg:25830", srs)

    def test_read_geometries_bbox_filter(self):
        """
        Check geometry filtering
        :return:
        """
        # get the bbox from the raster:
        raster_file = tcfg.resource("aerial1.tif")
        with rasterio.open(raster_file, 'r') as r:
            bbox = r.bounds
            bbox_filter = [bbox.left, bbox.bottom, bbox.right, bbox.top]

            feature_list, _ = read_features(tcfg.resource("test_parcels2/test_parcels2.shp"), bbox_filter)
        self.assertIsNotNone(feature_list)
        self.assertEqual(9, len(feature_list))

    def test_read_geometries_contains_filter(self):
        """
        Check geometry filtering
        :return:
        """
        # get the bbox from the raster:
        raster_file = tcfg.resource("aerial1.tif")
        feature_options = {"filter_type": "contains"}
        with rasterio.open(raster_file, 'r') as r:
            bbox = r.bounds
            bbox_filter = [bbox.left, bbox.bottom, bbox.right, bbox.top]

            feature_list, _ = read_features(tcfg.resource("test_parcels2/test_parcels2.shp"), bbox_filter,
                                            feature_options=feature_options)
        self.assertIsNotNone(feature_list)
        self.assertEqual(8, len(feature_list))

    def test_burn_features(self):
        raster_file = tcfg.resource("aerial1.tif")
        gdf = gpd.read_file(tcfg.resource("test_parcels2/test_parcels2.shp"))
        geo_list = gdf.geometry.tolist()
        out_folder = tempfile.gettempdir()

        out_f = burn_features(raster_file, geo_list, out_folder)

        os.path.exists(out_f)

    def test_mask_features(self):
        raster_file = tcfg.resource("aerial1.tif")
        gdf = gpd.read_file(tcfg.resource("test_parcels2/test_parcels2.shp"))
        geo_list = gdf.geometry.tolist()
        out_folder = tempfile.gettempdir();

        out_f = mask_features(raster_file, geo_list, out_folder)

        os.path.exists(out_f)

    def test_create_rectangles(self):
        raster_file = tcfg.resource("aerial1.tif")
        gdf = gpd.read_file(tcfg.resource("test_parcels2/test_parcels2.shp"))

        dataset = rasterio.open(raster_file)
        bbox = dataset.bounds
        dataset.close()
        # return poly
        filter = [bbox.left, bbox.bottom, bbox.right, bbox.top]
        feature_list, _ = read_features(tcfg.resource("test_parcels2/test_parcels2.shp"), filter)

        out_folder = tempfile.gettempdir()
        paths = create_rectangles(raster_file, feature_list, out_folder)

        self.assertIsNotNone(paths)
        self.assertEqual(len(feature_list), len(paths))


if __name__ == "__main__":
    unittest.main()
