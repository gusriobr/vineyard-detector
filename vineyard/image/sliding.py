import numpy as np


def sliding_window(image, window_size=(32, 32), step_size=5, batch_size=None, treat_borders=True):
    """
    creates sequences of images from ana original input image sliding a window across the input extracting patches
    :param image: input image
    :param step_size:  stride size, same in w and h direction
    :param window_size: (h,w) of the output images
    :param: batch_size: if set, in each iteration a batch of images of the passed size is retured
    :param treat_borders: if True, when the input image size is not a multiple of the patch, at the end of each row, the
     path is moved to the last position to return an image with the remaining pixels.
    :return: image of "window_size" shape
    """
    # treat image - patch size differences
    height, width, x_values, y_values = _calc_iter_ranges(image, step_size, treat_borders, window_size)

    for y in y_values:
        for x in x_values:
            yield image[y:y + height, x:x + width, :], x, y


def batched_sliding_window(image, window_size=(32, 32), step_size=5, batch_size=32, treat_borders=True):
    """
    creates sequences of images from ana original input image sliding a window across the input extracting patches
    :param image: input image
    :param step_size:  stride size, same in w and h direction
    :param window_size: (h,w) of the output images
    :param: batch_size: if set, in each iteration a batch of images of the passed size is retured
    :param treat_borders: if True, when the input image size is not a multiple of the patch, at the end of each row, the
     path is moved to the last position to return an image with the remaining pixels.
    :return: numpy array of images with shape (batch_size, window_size), list of positions (x,y)
    """
    # treat image - patch size differences
    height, width, x_values, y_values = _calc_iter_ranges(image, step_size, treat_borders, window_size)

    batch_shape = (batch_size,) + window_size
    batch_shape = batch_shape if len(image.shape) == 2 else (batch_size,) + window_size + (image.shape[2],)

    batch = np.zeros(batch_shape, dtype=image.dtype)
    positions = []
    i = 0
    for y in y_values:
        for x in x_values:
            batch[i, :] = image[y:y + height, x:x + width, :]
            positions.append((x, y))
            i += 1
            if i == batch_size:
                yield batch, positions
                # reset batch
                batch = np.zeros(batch_shape, dtype=image.dtype)
                positions = []
                i = 0
    if 0 < i < batch_size:
        # cut ending part of the batch with no values
        batch = batch[0:i, :].copy()
        positions = positions[0:i]
        yield batch, positions


def _calc_iter_ranges(image, step_size, treat_borders, window_size):
    height, width = window_size[0:2]
    if width > image.shape[0]:
        width = image.shape[0]
    if height > image.shape[1]:
        height = image.shape[1]
    # if image and patch size are the same, at least one image must be returned
    if height == image.shape[0]:
        y_values = [0]
    else:
        y_values = list(range(0, image.shape[0] - height + 1, step_size))
    if width == image.shape[1]:
        x_values = [0]
    else:
        x_values = list(range(0, image.shape[1] - width + 1, step_size))
    if treat_borders and y_values[-1] + height != image.shape[0]:
        y_values.append(image.shape[0] - height)
    if treat_borders and x_values[-1] + width != image.shape[1]:
        x_values.append(image.shape[1] - width)
    return height, width, x_values, y_values
