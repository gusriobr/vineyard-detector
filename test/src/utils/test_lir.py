import unittest

import math

from shapely import geometry
import pyproj
from shapely.geometry import shape
from matplotlib import pyplot as plt
from shapely.ops import transform

import numpy as np

from vineyard.utils import lir


# %%

def getCellMatrix(geom, size):
    minx, miny, maxx, maxy = geom.bounds
    width = round(maxx - minx)
    height = round(maxy - miny)
    nWidth = int(math.floor(width / size))
    nHeight = int(math.floor(height / size))

    grid = np.zeros((nHeight, nWidth), np.uint8)

    for i in range(nHeight):
        for j in range(nWidth):
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


def project_geo(geo, s_crs, d_crs):
    origin_crs = pyproj.CRS(s_crs)
    dest_crs = pyproj.CRS(d_crs)

    project = pyproj.Transformer.from_crs(origin_crs, dest_crs, always_xy=True).transform
    return transform(project, geo)


class TestLIR(unittest.TestCase):

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

    def test_lir(self):
        cells = np.array([[0, 0, 1, 0, 0, 0, 0, 0, 0],
                          [0, 0, 1, 0, 1, 1, 0, 0, 0],
                          [0, 0, 1, 1, 1, 1, 1, 0, 0],
                          [0, 0, 1, 1, 1, 1, 1, 1, 0],
                          [0, 0, 1, 1, 1, 1, 1, 1, 0],
                          [0, 1, 1, 1, 1, 1, 1, 0, 0],
                          [0, 0, 1, 1, 1, 1, 0, 0, 0],
                          [0, 0, 1, 1, 1, 1, 0, 0, 0],
                          [1, 1, 1, 1, 1, 1, 0, 0, 0],
                          [1, 1, 0, 0, 0, 1, 1, 1, 1],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0]])
        cells = np.uint8(cells * 255)

        h = lir.horizontal_adjacency(cells)
        v = lir.vertical_adjacency(cells)
        span_map = lir.span_map(h, v)
        rect = lir.biggest_span_in_span_map(span_map)
        rect2 = lir.largest_interior_rectangle(cells)
        plt.imshow(cells[rect2[1]:rect2[1] + rect2[3] + 1, rect2[0]:rect2[0] + rect2[2] + 1])

        np.testing.assert_array_equal(rect, np.array([2, 2, 4, 7]))
        np.testing.assert_array_equal(rect, rect2)

    def test_spans(self):
        cells = np.array([[1, 1, 1],
                          [1, 1, 0],
                          [1, 0, 0],
                          [1, 0, 0],
                          [1, 0, 0],
                          [1, 1, 1]])
        cells = np.uint8(cells * 255)

        h = lir.horizontal_adjacency(cells)
        v = lir.vertical_adjacency(cells)
        v_vector = lir.v_vector(v, 0, 0)
        h_vector = lir.h_vector(h, 0, 0)
        spans = lir.spans(h_vector, v_vector)

        np.testing.assert_array_equal(v_vector, np.array([6, 2, 1]))
        np.testing.assert_array_equal(h_vector, np.array([3, 2, 1]))
        np.testing.assert_array_equal(spans, np.array([[3, 1],
                                                       [2, 2],
                                                       [1, 6]]))

    def test_vector_size(self):
        t0 = np.array([1, 1, 1, 1], dtype=np.uint32)
        t1 = np.array([1, 1, 1, 0], dtype=np.uint32)
        t2 = np.array([1, 1, 0, 1, 1, 0], dtype=np.uint32)
        t3 = np.array([0, 0, 0, 0], dtype=np.uint32)
        t4 = np.array([0, 1, 1, 1], dtype=np.uint32)
        t5 = np.array([], dtype=np.uint32)

        self.assertEqual(lir.predict_vector_size(t0), 4)
        self.assertEqual(lir.predict_vector_size(t1), 3)
        self.assertEqual(lir.predict_vector_size(t2), 2)
        self.assertEqual(lir.predict_vector_size(t3), 0)
        self.assertEqual(lir.predict_vector_size(t4), 0)
        self.assertEqual(lir.predict_vector_size(t5), 0)



def starttest():
    unittest.main()


if __name__ == "__main__":
    starttest()

# README PLOTS

# from matplotlib import pyplot as plt

# data.py = np.flip(cells, 0)

# # The normal figure
# fig = plt.figure(figsize=(16, 12))
# ax = fig.add_subplot(111)
# ax.axes.xaxis.set_visible(False)
# ax.axes.yaxis.set_visible(False)
# im = ax.imshow(data.py, origin='lower', interpolation='None', cmap='Greys_r')
# fig.colorbar(im)

# data.py = np.flip(h, 0)

# # The normal figure
# fig = plt.figure(figsize=(16, 12))
# ax = fig.add_subplot(111)
# ax.axes.xaxis.set_visible(False)
# ax.axes.yaxis.set_visible(False)
# im = ax.imshow(data.py, origin='lower', interpolation='None', cmap='viridis')
# fig.colorbar(im)

# data.py = np.flip(v, 0)

# # The normal figure
# fig = plt.figure(figsize=(16, 12))
# ax = fig.add_subplot(111)
# ax.axes.xaxis.set_visible(False)
# ax.axes.yaxis.set_visible(False)
# im = ax.imshow(data.py, origin='lower', interpolation='None', cmap='viridis')
# fig.colorbar(im)

# data.py = np.flip(v*h, 0)

# h_vec = lir.h_vector(h, 2, 2)
# v_vec = lir.v_vector(v, 2, 2)
# spans = lir.spans(h_vec, v_vec)

# data.py = np.flip(span_map[:, :, 0], 0)

# # The normal figure
# fig = plt.figure(figsize=(16, 12))
# ax = fig.add_subplot(111)
# ax.axes.xaxis.set_visible(False)
# ax.axes.yaxis.set_visible(False)
# im = ax.imshow(data.py, origin='lower', interpolation='None', cmap='viridis')
# fig.colorbar(im)


# data.py = np.flip(span_map[:, :, 1], 0)

# # The normal figure
# fig = plt.figure(figsize=(16, 12))
# ax = fig.add_subplot(111)
# ax.axes.xaxis.set_visible(False)
# ax.axes.yaxis.set_visible(False)
# im = ax.imshow(data.py, origin='lower', interpolation='None', cmap='viridis')
# fig.colorbar(im)


# data.py = np.flip(span_map[:, :, 0] * span_map[:, :, 1], 0)

# # The normal figure
# fig = plt.figure(figsize=(16, 12))
# ax = fig.add_subplot(111)
# ax.axes.xaxis.set_visible(False)
# ax.axes.yaxis.set_visible(False)
# im = ax.imshow(data.py, origin='lower', interpolation='None', cmap='viridis')
# fig.colorbar(im)
