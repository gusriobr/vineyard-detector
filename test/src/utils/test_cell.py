import unittest

import geopandas as gpd
import rasterio
from matplotlib import pyplot as plt
from rasterio import features
from shapely.geometry import shape

import test.cfg_test as tcfg

from vineyard.utils import lir
from vineyard.data.patches import getCellMatrix, project_geo


class TestInnerRect(unittest.TestCase):

    def test_geo(self):
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

    def test_rasterize(self):
        shp_fn = tcfg.resource("test_parcels2/test_parcels2.shp")
        rst_fn = tcfg.resource("aerial1.tif")
        out_fn = tcfg.results('rasterized.tif')

        parcels = gpd.read_file(shp_fn)
        rst = rasterio.open(rst_fn)
        meta = rst.meta.copy()
        meta.update(compress='lzw')

        with rasterio.open(out_fn, 'w+', **meta) as out:
            out_arr = out.read(1)

            # this is where we create a generator of geom, value pairs to use in rasterizing
            shapes = ((geom, 255) for geom in parcels.geometry)

            burned = features.rasterize(shapes=shapes, fill=0, out=out_arr, transform=out.transform)
            out.write_band(1, burned)


if __name__ == "__main__":
    unittest.main()
