from keras.models import load_model
import keras

# Create the visualization instance.
# All visualization classes accept a model and model-modifier, which, for example,
#     replaces the activation of last layer to linear function so on, in constructor.
import cfg

# Render
from evaluation.activations import plot_filters, visualize_filter, visualize_filters

model_file = cfg.results("iteration2/vgg19.model")
model = load_model(model_file)
model.summary()


loss, img = visualize_filters(model, "/tmp/images")
# keras.preprocessing.image.save_img("/tmp/0.png", img)