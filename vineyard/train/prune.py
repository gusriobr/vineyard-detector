import tensorflow as tf
import tensorflow_model_optimization as tfmot
from keras.models import load_model


# Helper function uses `prune_low_magnitude` to make only the
# Dense layers train with pruning.
import cfg


def apply_pruning_to_dense(layer):
    if isinstance(layer, tf.keras.layers.Dense):
        return tfmot.sparsity.keras.prune_low_magnitude(layer)
    return layer


def prune_model(model_file):
    base_model = load_model(model_file)

    # model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(model)

    model_for_pruning = tf.keras.models.clone_model(
        base_model,
        clone_function=apply_pruning_to_dense,
    )

    # apply prunning to dense layers
    model_for_pruning.summary()


if __name__ == '__main__':
    model_file = cfg.results("model_first_try_best.model")
    prune_model(model_file)