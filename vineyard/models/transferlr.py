import tensorflow as tf

# from keras.applications.resnet import ResNet50
# from keras.applications.vgg19 import VGG19
import cfg

layers = tf.keras.layers

cfg.configLog()

resize = tf.keras.layers.Resizing(144, 144)


def build_binary_classifier(activation="sigmoid", augmentation_layer=None, freeze_weights=True, image_size=48):
    inputs = tf.keras.layers.Input(shape=(image_size, image_size, 3))
    x = inputs
    # x = resize(x)
    if augmentation_layer is not None:
        x = augmentation_layer(x)
    # trained_model = EfficientNetB0(include_top=False, input_tensor=x, weights="imagenet", drop_connect_rate=0.4)
    # trained_model = ResNet50V2(include_top=False, input_tensor=x, weights="imagenet")
    trained_model = tf.keras.applications.vgg19.VGG19(include_top=False, input_tensor=x, weights="imagenet")

    # Freeze the pretrained weights
    if freeze_weights:
        trained_model.trainable = False

    # Rebuild top
    x = layers.GlobalAveragePooling2D(name="avg_pool")(trained_model.output)
    x = layers.BatchNormalization()(x)

    top_dropout_rate = 0.2
    x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    outputs = layers.Dense(1, activation=activation, name="pred")(x)  # binary classification

    # Compile
    model = tf.keras.Model(inputs, outputs, name="VGG19")

    return model


def build_model_f(name, img_augmentation, image_size):
    def model_f():
        return build_model(name, activation="relu", augmentation_layer=img_augmentation, freeze_weights=True,
                           image_size=image_size)

    return model_f


def build_model(name, activation="relu", augmentation_layer=None, freeze_weights=True, image_size=48):
    inputs = tf.keras.layers.Input(shape=(image_size, image_size, 3), )
    x = inputs
    if augmentation_layer is not None:
        x = augmentation_layer(x)

    x = tf.keras.layers.Resizing(144, 144)(x)

    if name == "EfficientNetB0":
        trained_model = tf.keras.applications.efficientnet.EfficientNetB0(include_top=False, input_tensor=x,
                                                                          drop_connect_rate=0.4)
    if name == "InceptionV3":
        trained_model = tf.keras.applications.inception_v3.InceptionV3(include_top=False, input_tensor=x)
    if name == "ResNet50":
        trained_model = tf.keras.applications.resnet50.ResNet50(include_top=False, input_tensor=x)
    if name == "Xception":
        trained_model = tf.keras.applications.Xception(include_top=False, input_tensor=x)

    if freeze_weights:
        trained_model.trainable = False

    # Rebuild top
    x = layers.GlobalAveragePooling2D(name="avg_pool")(trained_model.output)
    x = layers.BatchNormalization()(x)

    top_dropout_rate = 0.3
    x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    outputs = layers.Dense(1, activation=activation, name="pred")(x)  # binary classification

    # Compile
    model = tf.keras.Model(inputs, outputs, name=name)

    return model


def build_binary_classifier_EffNetB0(activation="sigmoid", augmentation_layer=None, freeze_weights=True, image_size=48):
    inputs = tf.keras.layers.Input(shape=(image_size, image_size, 3))
    x = inputs
    # x = resize(inputs)
    if augmentation_layer is not None:
        x = augmentation_layer(x)
    trained_model = tf.keras.applications.efficientnet.EfficientNetB0(include_top=False, input_tensor=x,
                                                                      drop_connect_rate=0.4)
    # trained_model = ResNet50V2(include_top=False, input_tensor=x, weights="imagenet")
    # trained_model = tf.keras.applications.resnet.ResNet50(include_top=False, input_tensor=x, classes=1)

    # Freeze the pretrained weights
    if freeze_weights:
        trained_model.trainable = False

    # Rebuild top
    x = layers.GlobalAveragePooling2D(name="avg_pool")(trained_model.output)
    x = layers.BatchNormalization()(x)

    top_dropout_rate = 0.3
    x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    outputs = layers.Dense(1, activation=activation, name="pred")(x)  # binary classification

    # Compile
    model = tf.keras.Model(inputs, outputs, name="EfficientNetB0")

    return model


if __name__ == '__main__':
    build_binary_classifier()
