import tensorflow as tf

tf.config.set_visible_devices([], 'GPU')

import cfg
from evaluation.activations import visualize_filters

if __name__ == '__main__':
    ## visualize VGG16 filters
    image_size = 128
    intput_tensor = tf.keras.layers.Input(shape=(image_size, image_size, 3))
    model = tf.keras.applications.vgg19.VGG19(include_top=False, input_tensor=intput_tensor, weights="imagenet")
    output_vis = cfg.resource("vgg19_vis")
    visualize_filters(model, output_vis, image_size=image_size, suffix="{}_".format(image_size))
