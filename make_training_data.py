import os

import cv2
import matplotlib.pyplot as plt
import numpy as np


class Denoising():
    NOISED = "X:/dataset/noised"
    CLEAN = "X:/dataset/clean"
    TESTING = ""
    training_data = []

    def make_training_data(self):
        directory_n = self.NOISED
        directory_c = self.CLEAN
        for filename in os.listdir(directory_n):
            if filename.endswith(".jpg"):
                path_noised = os.path.join(directory_n, filename)
                img_noised = cv2.imread(path_noised)
                path_clean = os.path.join(directory_c, filename)
                img_clean = cv2.imread(path_clean)
                self.training_data.append([np.array(img_noised), np.array(img_clean)])  # do something like print(np.eye(2)[1]), just makes one_hot

        np.random.shuffle(self.training_data)
        np.save("training_data.npy", self.training_data)

dn = Denoising()
dn.make_training_data()