import numpy as np
import rasterio
from affine import Affine
from rasterio.windows import Window


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




#
# def standarize_dataset(x_test, mean, std):
#     x_tr = x_test * (1.0 / 255.0)
#     x_tr -= mean
#     x_tr /= std
#     return x_tr


if __name__ == '__main__':
    import rasterio as rio


    #
    # FUNCTIONS
    #
    def get_image_and_profile(path):
        with rio.open(path) as src:
            profile = src.profile
            w_transform = src.window_transform
            image = src.read()
        return image, profile, w_transform


    def image_write(im, path, profile):
        with rio.open(path, 'w', **profile) as dst:
            dst.write(im)


    def crop_image(image, crp):
        return image[:, crp:-crp, crp:-crp]


    def crop_profile(profile, img_shape, crp, w_transform):
        profile = profile.copy()
        profile.pop('transform', None)
        profile['width'] = img_shape[2]
        profile['height'] = img_shape[1]
        win = Window(crp, crp,  img_shape[2], img_shape[1])
        profile['transform'] = w_transform(win)
        return profile


    #
    # CROP GEOTIFF
    #
    path = '/test/resources/aerial1.tif'
    out_path = '/tmp/salidas/crop.tif'
    crp = 256

    im, profile, w_transform = get_image_and_profile(path)
    im = crop_image(im, crp)
    profile = crop_profile(profile, im.shape, crp, w_transform)
    image_write(im, out_path, profile)

    print("finished")