import scipy
from glob import glob
from keras.datasets import cifar10
import numpy as np
import matplotlib.pyplot as plt
import os
class DataLoader():
    def __init__(self, dataset_name, img_res=(32, 32)):
        self.dataset_name = dataset_name
        self.img_res = img_res

    def load_data(self, batch_size=1, is_testing=False, is_pred=False):
        data_type = "train" if not is_testing else "test"
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()
        if is_pred:
            imgs_hr = X_test[:batch_size]
        else:
            imgs_hr = X_train[np.random.choice(X_train.shape[0], size=batch_size)]

        return np.array(imgs_hr)


    def imread(self, path):
        return scipy.misc.imread(path, mode='RGB').astype(np.float)
