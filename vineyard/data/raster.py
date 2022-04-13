import rasterio
from affine import Affine


# import the necessary packages

def georeference_image(img, img_source, img_filename, scale=1, bands=3):
    """
    Creates a tiff raster to store the img param using geolocation
    information from another raster.

    :param img: Numpy array with shape [height, width, channels]
    :param img_source: raster to take the geolocation information from.
    :param img_filename: output raster filename
    :param scale: scale rate to apply to output image
    :return:
    """

    with rasterio.Env():
        # read profile info from first file
        dataset = rasterio.open(img_source)
        meta = dataset.meta.copy()
        dataset.close()

        meta.update({"driver": "GTiff", "count": bands, 'dtype': 'uint8'})
        meta.update({"width": img.shape[1], "height": img.shape[0]})
        new_affine = meta["transform"] * Affine.scale(1 / scale, 1 / scale)
        meta.update({"transform": new_affine})

        print(meta)

        with rasterio.open(img_filename, 'w', **meta, compress="JPEG") as dst: #, photometric="YCBCR"
            for ch in range(img.shape[-1]):
                # iterate over channels and write bands
                img_channel = img[:, :, ch]
                dst.write(img_channel, ch + 1)  # rasterio bands are 1-indexed


def standarize_dataset(x_test, mean, std):
    x_tr = x_test * (1.0 / 255.0)
    x_tr -= mean
    x_tr /= std
    return x_tr
