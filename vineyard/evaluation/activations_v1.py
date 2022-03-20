import datetime
import math
import os
import time
import tensorflow as tf

import matplotlib.cm as cm
import numpy as np
from keras import activations
from keras import backend as K
from keras.models import load_model
from keras.preprocessing.image import save_img
from matplotlib import pyplot as plt
from vis.input_modifiers import Jitter
from vis.utils import utils
from vis.visualization import overlay
from vis.visualization import visualize_activation, visualize_cam

from cnn.net import get_conv_layers_idx, get_conv_layer_by_name


def get_grid_dim(x):
    tf.image.psnr
    """
    Transforms x into product of two integers
    :param x: int
    :return: two ints
    """
    factors = prime_powers(x)
    if len(factors) % 2 == 0:
        i = int(len(factors) / 2)
        return factors[i], factors[i - 1]

    i = len(factors) // 2
    return factors[i], factors[i]


def prime_powers(n):
    """
    Compute the factors of a positive integer
    Algorithm from https://rosettacode.org/wiki/Factors_of_an_integer#Python
    :param n: int
    :return: set
    """
    factors = set()
    for x in range(1, int(math.sqrt(n)) + 1):
        if n % x == 0:
            factors.add(int(x))
            factors.add(int(n // x))
    return sorted(factors)


def plot_images(images, cls_true, cls_pred=None, img_shape = None, grid=(3, 3)):
    assert len(images) == len(cls_true) == 9
    plt.figure()
    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(grid[0], grid[1])
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(images[i].reshape(img_shape), cmap='binary')

        # Show true and predicted classes.
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])

        # Show the classes as the label on the x-axis.
        ax.set_xlabel(xlabel)

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    return plt


def plot_conv_weights(model, layer_idx, input_channel="all", output_folder=None):
    """
    plot_conv_weights(model,0, output_folder="/tmp")

       Plots convolutional filters
        :param model: reference model
        :param layer_idx: conv layer index in the model

       :param name: string, name of convolutional layer
       :param input_channel: channels selected of "all" to plot all filter channels
       :param plot_dir: folder to store images in case "all" option is used in "input_channel" parameter
       :return: nothing, plots are saved on the disk
       """

    layer = model.layers[layer_idx]
    if not "conv" in layer.__class__.__name__.lower():
        raise Exception("The parameter idx doesn't refer to a Convolution layer: {}.".format(layer.__class__.__name__))
    weights = layer.get_weights()[0]

    # Get the lowest and highest values for the weights.
    # This is used to correct the colour intensity across
    # the images so they can be compared with each other.
    w_min = np.min(weights)
    w_max = np.max(weights)

    # Number of filters used in the conv. layer.
    num_filters = weights.shape[3]

    # Number of grids to plot.
    # Rounded-up, square-root of the number of filters.
    num_grids = math.ceil(math.sqrt(num_filters))

    channels = [input_channel]
    # make a list of channels if all are plotted
    if input_channel == "all":
        channels = range(weights.shape[2])

    ts = datetime.datetime.now().timestamp()

    # iterate channels
    for channel in channels:
        # Create figure with a grid of sub-plots.
        fig, axes = plt.subplots(num_grids, num_grids)
        # Plot all the filter-weights.
        for i, ax in enumerate(axes.flat):
            # Only plot the valid filter-weights.
            if i < num_filters:
                # Get the weights for the i'th filter of the input channel.
                # See new_conv_layer() for details on the format
                # of this 4-dim tensor.
                img = weights[:, :, channel, i]

                # Plot image.
                ax.imshow(img, vmin=w_min, vmax=w_max,
                          interpolation='nearest', cmap='seismic')

            # Remove ticks from the plot.
            ax.set_xticks([])
            ax.set_yticks([])
        if input_channel == "all":
            plt.savefig(os.path.join(output_folder, 'conv_weights_{}_{}.png'.format(ts, channel)), bbox_inches='tight')

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    return plt


def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + K.epsilon())
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_data_format() == 'channels_first':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + K.epsilon())


def plot_conv_output(model, image_size=(128, 128), layer_filter=[], max_filters=32, output_folder="."):
    img_height, img_width = image_size
    kept_filters = []

    # if no layer filter is defined, get all conv layers from the model
    selected_layers = []
    if layer_filter is None or not layer_filter:
        # select all conv. layers
        selected_layers = [l.name for l in model.layers if "Conv2D" in str(l.__class__)]
    else:
        selected_layers = [l.name for l in model.layers if any(x in l.name for x in layer_filter)]

    # this is the placeholder for the input images
    input_img = model.input

    # get the symbolic outputs of each "key" layer (we gave them unique names).
    layer_dict = dict([(layer.name, layer) for layer in model.layers])

    for layer_name in selected_layers:
        print('____Processing layer %s_____' % layer_name)
        for filter_index in range(max_filters):
            # we only scan through the first 200 filters,
            # but there are actually 512 of them
            print('Processing filter %d' % filter_index)
            start_time = time.time()

            # we build a loss function that maximizes the activation
            # of the nth filter of the layer considered
            layer_output = layer_dict[layer_name].output
            if K.image_data_format() == 'channels_first':
                loss = K.mean(layer_output[:, filter_index, :, :])
            else:
                loss = K.mean(layer_output[:, :, :, filter_index])

            # we compute the gradient of the input picture wrt this loss
            grads = K.gradients(loss, input_img)[0]

            # normalization trick: we normalize the gradient
            grads = normalize(grads)

            # this function returns the loss and grads given the input picture
            iterate = K.function([input_img], [loss, grads])

            # we start from a gray image with some random noise
            if K.image_data_format() == 'channels_first':
                input_img_data = np.random.random((1, 3, img_width, img_height))
            else:
                input_img_data = np.random.random((1, img_width, img_height, 3))  # TODO: change image width/heigh
            input_img_data = (input_img_data - 0.5) * 20 + 128

            # step size for gradient ascent
            step = 1.
            # we run gradient ascent for 20 steps
            for i in range(20):
                loss_value, grads_value = iterate([input_img_data])
                input_img_data += grads_value * step

                # print('Current loss value:', loss_value)
                if loss_value <= 0.:
                    # some filters get stuck to 0, we can skip them
                    break

            # decode the resulting input image
            if loss_value > 0:
                img = deprocess_image(input_img_data[0])
                kept_filters.append((img, loss_value))
            end_time = time.time()
            # print('Filter %d of %d processed in %ds' % (filter_index, max_filters, end_time - start_time))

        # we will stich the best 64 filters on a 8 x 8 grid.
        n = 7

        # the filters that have the highest loss are assumed to be better-looking.
        # we will only keep the top 64 filters.
        kept_filters.sort(key=lambda x: x[1], reverse=True)
        kept_filters = kept_filters[:n * n]

        # build a black picture with enough space for
        # our 8 x 8 filters of size 128 x 128, with a 5px margin in between
        margin = 5
        width = n * img_width + (n - 1) * margin
        height = n * img_height + (n - 1) * margin
        stitched_filters = np.zeros((width, height, 3))

        # fill the picture with our saved filters
        for i in range(n):
            for j in range(n):
                if i * n + j >= len(kept_filters):
                    break
                img, loss = kept_filters[i * n + j]
                width_margin = (img_width + margin) * i
                height_margin = (img_height + margin) * j
                stitched_filters[
                width_margin: width_margin + img_width,
                height_margin: height_margin + img_height, :] = img

        # save the result to disk
        save_img(os.path.join(output_folder, 'stitched_filters_%dx%d_%s.png' % (n, n, layer_name)), stitched_filters)


def find_last_dense_layer(model):
    layers = model.layers
    i = len(model.layers) - 1
    for l in reversed(layers):
        if "dense" in l.__class__.__name__.lower():
            return i
        i -= 1
    return -1


def plot_activation_dense(model, layer_idx=-1, categories="all", labels=None, output_file=None, max_iter=50,
                          verbose=False, input_range=(0, 1)):
    """
    Generates the conv activation images for a dense layer, typically used to see the images that maximize
    the final prediction layer.
    * plot_activation_dense(model): generates maximization images for all categories
    * plot_activation_dense(model,-1): generates maximization images for all categories
    * plot_activation_dense(model,output_file="/tmp/pred_activation.png"): generates maximization images for
        all categories and stores the resulting image in /tmp folder.
    * plot_activation_dense(model,-1, [1,2,3]): generates maximization images for categories 1,2,3
    * plot_activation_dense(model,-1, [1,2,3],labels=['dog','cat','car')): generates maximization images
    for categories 1,2,3 and set the proper labels in image text
    https://github.com/raghakot/keras-vis/blob/master/examples/vggnet/activation_maximization.ipynb
    :param model:
    :param max_iter:
    :param layer_idx: dense layer idx, for final predictions, use -1
    :param categories: integer or list indicating the categories to filter in the resulting image.
    :return:
    """
    # layer_idx = utils.find_layer_idx(model, 'predictions')
    # Swap softmax with linear
    if layer_idx == -1:
        layer_idx = find_last_dense_layer(model)

    if type(categories) != list:
        if str(categories) == "all":
            # all categories
            dense_layer = model.layers[layer_idx]
            categories = list(range(0, dense_layer.output_shape[-1]))
        else:
            categories = [categories]

    vis_images = []
    image_modifiers = [Jitter(16)]

    model.layers[layer_idx].activation = activations.linear
    model = utils.apply_modifications(model)

    for idx in categories:
        # this has to explored depending on the problem, for car-log, 1 is the best option, to get
        # meaningfull images
        # for tv_weight in [1e-3, 1e-2, 1e-1, 1, 10]:
        tv_weight = 10
        img = visualize_activation(model, layer_idx, filter_indices=idx, max_iter=max_iter,
                                   input_range=(0., 1.), tv_weight=tv_weight, lp_norm_weight=0., verbose=True)
        # img = visualize_activation(model, layer_idx, filter_indices=idx, max_iter=max_iter,input_modifiers=image_modifiers)

        # Reverse lookup index to imagenet label and overlay it on the image.
        label = labels[idx] if labels is not None else "cat. {}".format(idx)
        # if image is bn change to rgb
        if img.shape[2] == 1:
            pass
            # img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            # img = img.reshape((img.shape[:2]))
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = (img * 255).astype(np.uint8)
        img = utils.draw_text(img, "{}_{}".format(label, tv_weight), position=(3, 3), font_size=11)
        vis_images.append(img)

    # Generate stitched images with 5 cols.
    plt.rcParams['figure.figsize'] = (50, 50)
    stitched = utils.stitch_images(vis_images, cols=5)
    plt.axis('off')
    plt.imshow(stitched)

    if output_file:
        plt.savefig(output_file)

    return plt


def plot_activation_filters(model, layer_idx, filters="all", output_folder=None,
                            max_iter=50, verbose=False, prefix = ""):
    """
    Plots the activation for selected filters in conv layers.
    Usage:
    * plot_activation_filters(model, 12): plots all filter for layer 12 (13th) in the model.
    * plot_activation_filters(model, 12, filters = [0,1,2,3]): plots the firsts four filters in the 13th layer.
    * plot_activation_filters(model, all, filters = *range(20)): plots the firsts 20 filters in each conv. layer.
    :param model:
    :param layer_idx: conv layer index in the model. If all is passed, all conv layers are used.
    :param filters:  selected filters for the conv. layer. If "all" is passed, al filters are taken in account (considered).
    :param output_file: file to store resulting image.
    :param max_iter:
    :param verbose:
    :param prefix: prefix to preceed the name of the the resulting images
    :return:
    """

    if layer_idx == "all":
        conv_layers = get_conv_layers_idx(model)
    else:
        if type(layer_idx) != list:
            conv_layers = [layer_idx]
        else:
            conv_layers = layer_idx

    # iterate over conv_layers array replacing layer names by their index
    cv = []
    for l in conv_layers:
        if type(l) == str:
            idx = get_conv_layer_by_name(model, [l])
            if len(idx) == 0:
                raise Exception("The conv layer {} couldn't be found.".format(l))
            idx = idx[0]
        else:
            idx = l
        cv.append(idx)
    conv_layers = cv

    if type(filters) != list:
        if str(filters) == "all":
            # all categories
            convlayer = model.layers[layer_idx]
            filters = list(range(0, convlayer.output_shape[-1]))
        else:
            filters = [filters]

    image_modifiers = [Jitter(16)]

    # model.layers[layer_idx].activation = activations.linear
    # model = utils.apply_modifications(model)

    for conv in conv_layers:
        vis_images = []
        #  make sure we don't try to get more filters that existing
        num_filters = model.layers[conv].output_shape[-1]
        if len(filters) > num_filters:
            selected_filters = list(range(num_filters))
        else:
            selected_filters = filters
        for idx in selected_filters:
            # this has to explored depending on the problem, for car-log, 1 is the best option, to get
            # meaningfull images
            # for tv_weight in [1e-3, 1e-2, 1e-1, 1, 10]:
            # tv_weight = 10
            # print ("Plotting activation form layer: {} filter {}".format(conv, idx))
            img = visualize_activation(model, conv, filter_indices=idx, max_iter=max_iter,
                                       # input_range=(0., 1.),
                                       # verbose=True,
                                       tv_weight=0.,
                                       input_modifiers=[Jitter(0.05)]
                                       )

            # if image is bn change to rgb
            if img.shape[2] == 1:
                pass
                # img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                # img = img.reshape((img.shape[:2]))
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # img = (img * 255).astype(np.uint8)
            img = utils.draw_text(img, str(idx), position=(3, 3), font_size=11)
            vis_images.append(img)

        # Generate stitched images with 5 cols.
        plt.rcParams['figure.figsize'] = (50, 50)
        stitched = utils.stitch_images(vis_images, cols=5)
        plt.figure()
        plt.axis('off')
        plt.imshow(stitched)
        if prefix != "":
            prefix = prefix + "_"

        plt.savefig(os.path.join(output_folder, "{}filters_{}_{}.png".format(prefix, conv, model.layers[conv].name)))

    return plt


def report_activations(models, labels=None, output_folder=None,
                       categories="none", model_name=None,
                       conv_layers="all", conv_layer_filters="all"):
    """
    Plots the convolution filters and predictions activation images for the given models.

    report_activations(my_models, my_labels): saves all filter for all conv layers in the passed models.
    report_activations(my_models, my_labels, categories=[*range(0, 10)]): saves all filter for all
        conv layers in the passed models and just the first 10 categories predictions images.
    report_activations(my_models, my_labels, conv_layers = [0,3,9]): saves all filter for the conv. layers
        0,3,9.
    report_activations(my_models, my_labels, conv_layers = "all", conv_layer_filters = list(range(0,15))): saves
        the first 15 filter for all conv. layers of the model.


    :param models: model file, file name, list of model files.
    :param labels: labels used in the model training. If None, the category index will be used.
    :param categories: none, all, list of integers. Filter to determine which categories must be
        plot on the predictions image.
    :param output_folder: folder to store resulting images
    :param conv_layers: list with the indexes of the conv. layers to used in the selected models.
    :param conv_layer_filters: filter to select the conv filters.
    :return:
    """
    if not output_folder:
        output_folder = "results"

    if type(models) == str:
        models = [models]

    if type(models) == list:
        for file in models:
            f_name = os.path.basename(file)
            model = load_model(file)
            report_model_activations(model, labels=labels, output_folder=output_folder, model_name=f_name,
                                     categories=categories, conv_layers=conv_layers,
                                     conv_layer_filters=conv_layer_filters)
    else:
        # just one model as parameter
        model = models
        report_model_activations(model, labels=labels, output_folder=output_folder,
                                 categories=categories, conv_layers=conv_layers,
                                 conv_layer_filters=conv_layer_filters)


def report_model_activations(model, labels=None, output_folder=None, model_name="model",
                             categories="none",
                             conv_layers="all", conv_layer_filters="all"):
    """
    Plots the convolution filters and predictions activation images for the given model.
    :param model: model to take in account
    :param labels: categories labels
    :param output_folder:
    :param model_name: name for the image output file.
    :param categories:
    :param conv_layers:
    :param conv_layer_filters:
    :return:
    """
    if categories != "none":
        file_path = os.path.join(output_folder, "activations.png")
        plot_activation_dense(model, output_file=file_path, labels=labels, categories=categories)

    if not type(conv_layers) == list:
        conv_layers = get_conv_layers_idx(model)

    for conv in conv_layers:
        file_path = os.path.join(output_folder, "filters_conv_{}.png".format(conv))
        plot_activation_filters(model, layer_idx=conv, filters=conv_layer_filters, output_file=file_path)


def get_activations(model, model_inputs, layer_name=None):
    activations = []
    inp = model.input

    model_multi_inputs_cond = True
    if not isinstance(inp, list):
        # only one input! let's wrap it in a list.
        inp = [inp]
        model_multi_inputs_cond = False

    outputs = [layer.output for layer in model.layers if
               layer.name == layer_name or layer_name is None]  # all layer outputs

    # we remove the placeholders (Inputs node in Keras). Not the most elegant though..
    outputs = [output for output in outputs if 'input_' not in output.name]

    funcs = [K.function(inp + [K.learning_phase()], [out]) for out in outputs]  # evaluation functions

    if model_multi_inputs_cond:
        list_inputs = []
        list_inputs.extend(model_inputs)
        list_inputs.append(0.)
    else:
        list_inputs = [model_inputs, 0.]

    # Learning phase. 0 = Test mode (no dropout or batch normalization)
    # layer_outputs = [func([model_inputs, 0.])[0] for func in funcs]
    activations = [func(list_inputs)[0] for func in funcs]
    layer_names = [output.name for output in outputs]

    result = dict(zip(layer_names, activations))
    return result


def display_activations(activations, output_folder="/tmp", prefix=""):
    """
    (1, 26, 26, 32)
    (1, 24, 24, 64)
    (1, 12, 12, 64)
    (1, 12, 12, 64)
    (1, 9216)
    (1, 128)
    (1, 128)
    (1, 10)
    """
    layer_names = list(activations.keys())
    activation_maps = list(activations.values())
    batch_size = activation_maps[0].shape[0]
    assert batch_size == 1, 'One image at a time to visualize.'
    for i, activation_map in enumerate(activation_maps):
        print('Displaying activation map {}'.format(i))
        shape = activation_map.shape
        if len(shape) == 4:
            activations = np.hstack(np.transpose(activation_map[0], (2, 0, 1)))
        elif len(shape) == 2:
            # try to make it square as much as possible. we can skip some activations.
            activations = activation_map[0]
            num_activations = len(activations)
            if num_activations > 1024:  # too hard to display it on the screen.
                square_param = int(np.floor(np.sqrt(num_activations)))
                activations = activations[0: square_param * square_param]
                activations = np.reshape(activations, (square_param, square_param))
            else:
                activations = np.expand_dims(activations, axis=0)
        else:
            raise Exception('len(shape) = 3 has not been implemented.')
        plt.figure()
        plt.title(layer_names[i])
        plt.imshow(activations, interpolation='None', cmap='jet')
        plt.savefig(os.path.join(output_folder, "{}_activations_{}.png".format(prefix, layer_names[i])))


def plot_cam(model, category, test_img, output_file, layer_idx=-1):
    if layer_idx == -1:
        layer_idx = find_last_dense_layer(model)
    # for modifier in [None, 'guided', 'relu']:
    grads = visualize_cam(model, layer_idx, filter_indices=category, seed_input=test_img,
                          backprop_modifier="guided")
    jet_heatmap = np.uint8(cm.jet(grads)[..., :3] * 255)
    plt.imshow(overlay(jet_heatmap, test_img))
    plt.savefig(output_file)



