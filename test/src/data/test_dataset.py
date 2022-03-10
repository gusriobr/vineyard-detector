import os
import pickle
import unittest

import test.cfg_test as tcfg
from vineyard.data.dataset import split_patches, load_images_as_array, load_batch, create_dataset_file, load


class TestDataSet(unittest.TestCase):

    def test_split_patches(self):
        patches_folder = tcfg.resource("patches")

        expected_train = int(10 * (1 - 0.3))
        expected_test = 10 - expected_train
        split_patches(patches_folder, "/tmp", 0.3)

        self.assertTrue(os.path.exists("/tmp/train/1"))
        self.assertTrue(os.path.exists("/tmp/train/2"))
        self.assertTrue(os.path.exists("/tmp/test/1"))
        self.assertTrue(os.path.exists("/tmp/test/2"))
        self.assertEqual(expected_train, len(os.listdir("/tmp/train/1")))
        self.assertEqual(expected_test, len(os.listdir("/tmp/test/1")))

    def test_load_images_as_array(self):
        folder = tcfg.resource("patches/1")
        arr = load_images_as_array(folder)
        self.assertEqual((10, 48, 48, 3), arr.shape)

    def test_load_batch(self):
        splits_folder = tcfg.resource("patches")
        x, y = load_batch(splits_folder)
        self.assertEqual((18, 48, 48, 3), x.shape)
        self.assertEqual((18, 1), y.shape)

    def test_create_dataset_file(self):
        patches_folder = tcfg.resource("patches")
        create_dataset_file(patches_folder, "/tmp")

        dataset_file = "/tmp/data.npy"
        self.assertTrue(os.path.exists(dataset_file))
        # Load data (deserialize)
        data = load(dataset_file)
        self.assertEqual(2, len(data))


if __name__ == "__main__":
    unittest.main()
