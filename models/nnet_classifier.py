from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

import pandas as pd
import io
import requests
import sys

"""
Example usage:
nnet = NNet()
    nnet.read_csv_data_from_url('http://download.tensorflow.org/data/iris_training.csv',
                                col_titles=['Sepal Length', 'Sepal Width',
                      'Petal Length', 'Petal Width', 'Species'],
                                labels='Species')
    nnet.train_model(model_directory='/Users/edwardcannon/git-repos/nnets/models',
                     hidden_units=[10,20,10],
                     num_classes=3)
    y_pred = nnet.classifier.predict(test_data, as_iterable=True)
    for val in y_pred:
        print(val)
    accuracy_score = classifier.evaluate(x=test_x,
                                         y=test_y)["accuracy"]

"""

__author__ = 'edwardcannon'


class NNet(object):
    """
    Simple neural network classifier
    """
    def __init__(self):
        self.training_data = pd.DataFrame()
        self.train_labels = []
        self.classifier = None

    def read_csv_data_from_url(self, url, col_titles=[], labels='label'):
        """
        Extracts csv data from url returns
        pandas data frame
        :param url: url to extract csv data
        :param col_titles: header
        :param labels: labels y-column
        """
        data = requests.get(url).content
        df = pd.read_csv(io.StringIO(data.decode('utf-8')))
        if len(col_titles) != 0:
            df.columns = col_titles
        df_tr = df.drop(labels, 1)
        self.training_data = df_tr
        self.train_labels = df[labels]

    def train_model(self, model_directory, hidden_units =[],
                    num_classes=2, num_steps=2000):
        """
        Trains neural network with user defined hidden units
        :param model_directory: Output model directory
        :param hidden_units: Number of hidden units
        :param num_classes: Number of classes to train on
        :param num_steps: Number of steps in training
        :return:
        """
        feature_columns = [tf.contrib.layers.real_valued_column("",dimension=self.training_data.shape[1])]
        self.classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                                hidden_units=hidden_units,
                                                n_classes=num_classes,
                                                model_dir=model_directory)
        self.classifier.fit(x=self.training_data,
                            y=self.train_labels,
                            steps=num_steps)


if __name__ == '__main__':
    pass


