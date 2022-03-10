import joblib
import numpy as np
from keras.models import load_model
from keras_preprocessing.image import ImageDataGenerator

import cfg
from data import dataset
from data.model import DatasetDef
from evaluation.charts import plot_model_chart

def evaluate_model(model, base_name, x_test, y_test):
    _, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(accuracy)
    history_file = base_name.replace(".model", ".hist")
    history = joblib.load(history_file)
    plot_model_chart(history, base_name.replace(".model", "_chart.png"))

def evaluate_model_file(model_file, x_test, y_test):
    model = load_model(model_file)
    evaluate_model(model, model_file, x_test, y_test)


if __name__ == '__main__':
    dataset_file = "/media/gus/data/viticola/dataset/dataset.npy"

    (x_train, y_train), (x_test, y_test) = dataset.load(dataset_file)
    model_file = cfg.results("model_first_try_best.model")
    # create el generador de datos para la conv. de las imagenes
    datagen = ImageDataGenerator(rescale=1. / 255, featurewise_center=True, featurewise_std_normalization=True)
    model_conf = DatasetDef.read(model_file.replace(".model", ".json"))
    datagen.mean = model_conf["mean"]
    datagen.std = model_conf["std"]

    # better do it manually
    # predict_gen = datagen.flow(x_test, y=None, shuffle=False, batch_size=len(y_test))
    # x_tr = predict_gen.next()

    x_tr2 = np.zeros(shape=x_test.shape, dtype=np.float32)
    x_tr2 = x_test * (1.0 / 255.0)
    x_tr2 -= model_conf["mean"]
    x_tr2 /= model_conf["std"]

    evaluate_model(model_file, x_tr2, y_test)
