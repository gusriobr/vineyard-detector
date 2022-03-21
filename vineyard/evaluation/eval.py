import logging
import os
from pathlib import Path

import joblib
import pandas as pd
import tensorflow as tf

import cfg
from evaluation.charts import plot_model_chart

cfg.configLog()


def evaluate_model(model, model_file, x_test, y_test):
    _, accuracy = model.evaluate(x_test, y_test, verbose=0)
    logging.info("Evaluating model " + model_file)
    logging.info("model accuracy = {}".format(accuracy))
    history_file = os.path.join(Path(model_file).parent.absolute(), "history.jbl")
    history = joblib.load(history_file)
    plot_model_chart(history, model_file.replace(".md", "_chart.png"))


def evaluate_models(mfile_list, x_test, y_test):
    # free memory
    tf.keras.backend.clear_session()
    for file in mfile_list:
        logging.info("Evaluating model " + file)
        evaluate_model_file(file, x_test, y_test)


def evaluate_model_file(model_file, x_test, y_test):
    model = tf.keras.models.load_model(model_file)
    evaluate_model(model, model_file, x_test, y_test)


def summarize(base_folder):
    """
    Looks for model folders below base_folder directory to find the history.jbl files. For each file found, retrieves
    the max val_accurary for each model
    :param model_labels:
    :return:
    """
    # import tensorflow as tf
    # tf.config.set_visible_devices([], 'GPU')  # force CPU

    # find nested folders with the history.jbl file
    model_folders = [folder for folder in os.listdir(base_folder) if
                     os.path.exists(os.path.join(base_folder, folder, "history.jbl"))]
    best_values = list()
    for model_label in model_folders:
        logging.info("Reading history file " + model_label)

        history = joblib.load(os.path.join(base_folder, model_label, "history.jbl"))
        if hasattr(history, "history"):  # full keras history object create panda df
            hist = dict(history.history)
            if 'lr' in hist:
                hist.pop("lr")  # remove key
            df_hist = pd.DataFrame(hist)
        else:
            df_hist = history

        # get best value from history
        values = df_hist.sort_values(by="val_accuracy", ascending=False).iloc[0].to_dict()
        values["model"] = model_label
        best_values.append(values)

    df_summary = pd.DataFrame(best_values)
    # write as text
    with open(os.path.join(base_folder, "summary.txt"), 'a') as f:
        df_as_string = df_summary.to_string(header=True, index=False)
        f.write(df_as_string)


if __name__ == '__main__':
    base_output = cfg.results("iteration2")
    summarize(base_output)
