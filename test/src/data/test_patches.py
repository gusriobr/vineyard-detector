import unittest

from skimage.io import imread

import test.cfg_test as tcfg
from vineyard.data.patches import extract_patches, sliding_window


class TestPatches(unittest.TestCase):

    def test_extract_patches(self):
        base_folder = "/tmp/patches"
        extract_patches(tcfg.resource("lir"), base_folder, patch_options={"folder_per_category": True, "size": 48})

    def test_sliding_window(self):
        img = imread(tcfg.resource("lir/1_93BFED23EE.png"))

        patches = list(sliding_window(img))

        self.assertGreater(len(patches), 0)


if __name__ == "__main__":
    unittest.main()
