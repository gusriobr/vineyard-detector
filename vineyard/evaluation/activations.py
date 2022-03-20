import os.path
from pathlib import Path

import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

"""
Based on https://keras.io/examples/vision/visualizing_what_convnets_learn/
"""


def deprocess_image(img):
    # Normalize array: center on 0., ensure variance is 0.15
    img -= img.mean()
    img /= img.std() + 1e-5
    img *= 0.15

    # Center crop
    # img = img[25:-25, 25:-25, :]

    # Clip to [0, 1]
    img += 0.5
    img = np.clip(img, 0, 1)

    # Convert to RGB array
    img *= 255
    img = np.clip(img, 0, 255).astype("uint8")
    return img


def plot_filters(model, layers_names=None):
    if layers_names is not None:
        layers_names = [l.lower() for l in layers_names]

    for layer in model.layers:
        if layers_names is not None and layer.name.lower() not in layers_names:
            continue

        if 'conv' not in layer.name.lower():
            continue

        weights, bias = layer.get_weights()

        # normalize filter values between  0 and 1 for visualization
        f_min, f_max = weights.min(), weights.max()
        filters = (weights - f_min) / (f_max - f_min)
        print(filters.shape[3])
        filter_cnt = 1

        # plotting all the filters
        for i in range(filters.shape[3]):
            # get the filters
            filt = filters[:, :, :, i]
            # plotting each of the channel, color image RGB channels
            for j in range(filters.shape[0]):
                ax = plt.subplot(filters.shape[3], filters.shape[0], filter_cnt)
                ax.set_xticks([])
                ax.set_yticks([])
                plt.imshow(filt[:, :, j])
                filter_cnt += 1
        plt.show()

        plt.savefig("/tmp/salida.png")


def initialize_image(img_width=48, img_height=48):
    # We start from a gray image with some random noise
    img = tf.random.uniform((1, img_width, img_height, 3))
    # ResNet50V2 expects inputs in the range [-1, +1].
    # Here we scale our random inputs to [-0.125, +0.125]
    return (img - 0.5) * 0.25


def compute_loss(model, layer_name, input_image, filter_index):
    layer = model.get_layer(name=layer_name)
    # create a model using as input the image and as output the desired layer's output
    feature_extractor = keras.Model(inputs=model.inputs, outputs=layer.output)
    activation = feature_extractor(input_image)
    # activation = layer(input_image)

    # We avoid border artifacts by only involving non-border pixels in the loss.
    filter_activation = activation[:, 2:-2, 2:-2, filter_index]
    rd = tf.reduce_mean(filter_activation)
    return rd


# @tf.function
def gradient_ascent_step(model, layer_name, img, filter_index, learning_rate):
    with tf.GradientTape() as tape:
        tape.watch(img)
        loss = compute_loss(model, layer_name, img, filter_index)
    # Compute gradients.
    grads = tape.gradient(loss, img)
    # Normalize gradients.
    grads = tf.math.l2_normalize(grads)
    img += learning_rate * grads
    return loss, img


def initialize_image(img_width=256, img_height=256):
    # We start from a gray image with some random noise
    img = tf.random.uniform((1, img_width, img_height, 3))
    # ResNet50V2 expects inputs in the range [-1, +1].
    # Here we scale our random inputs to [-0.125, +0.125]
    # return (img - 0.5) * 0.25
    return img


def visualize_filter(model, layer_name, filter_index=None, img_size=48):
    # We run gradient ascent for 20 steps
    iterations = 20
    learning_rate = 10.0
    img = initialize_image(img_size, img_size)

    for iteration in range(iterations):
        loss, img = gradient_ascent_step(model, layer_name, img, filter_index, learning_rate)

    # Decode the resulting input image
    img = deprocess_image(img[0].numpy())
    return loss, img


def plot_filter_activations(images, output_path, img_width, img_height, n=8):
    # Build a black picture with enough space for
    # our 8 x 8 filters of size 128 x 128, with a 5px margin in between
    margin = 5
    n = 8
    cropped_width = img_width  # - 25 * 2
    cropped_height = img_height  # - 25 * 2
    width = n * cropped_width + (n - 1) * margin
    height = n * cropped_height + (n - 1) * margin
    stitched_filters = np.zeros((width, height, 3))

    # Fill the picture with our saved filters
    for i in range(n):
        for j in range(n):
            index = i * n + j
            if index >= len(images):
                break
            img = images[index]
            stitched_filters[(cropped_width + margin) * i: (cropped_width + margin) * i + cropped_width,
            (cropped_height + margin) * j: (cropped_height + margin) * j
                                           + cropped_height, :, ] = img

    keras.preprocessing.image.save_img(output_path, stitched_filters)


def visualize_filters(model, output_folder, layer_names=None, image_size=48, max_filters=64, prefix="", suffix=""):
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    if layer_names is None:
        layer_names = [layer.name for layer in model.layers if hasattr(layer, "filters") and layer.filters is not None]
    for idx, layer_name in enumerate(layer_names):
        layer = model.get_layer(name=layer_name)
        all_imgs = []
        print("Processing layer {}".format(layer_name))
        for filter_index in range(layer.filters):
            if filter_index >= layer.filters or filter_index >= max_filters:
                break;
            print("Processing filter %d" % (filter_index,))
            loss, img = visualize_filter(model, layer_name, filter_index, img_size=image_size)
            all_imgs.append(img)

        output_path = os.path.join(output_folder,
                                   "{}{}_{}{}.png".format(prefix, idx, layer_name.replace(" ", "_"), suffix))
        plot_filter_activations(all_imgs, output_path, img_width=image_size, img_height=image_size)
