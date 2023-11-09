import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA 
from tensorflow.keras.models import Model
from tensorflow.keras import layers, losses
import pandas as pd
import os

"""
With the filename as an input, data is read from a csv file and pre-processed for both our QCNN and CNN models. This code is written specifically for analysis of medical dataset, where the train and test data splits has already been performed

filename: string detailing file location
classes: possible solutions to classification task
"""
def load_and_process(filename, classes=[0, 1], feature_reduction='resize256', binary=True):
    root = filename

    train_location = root + "/train"
    training_dataset = pd.read_csv(filename)

    return 0

def __main__():
    root = "data/chest_xray"

    print(os.listdir(root))

    return 1

__main__()