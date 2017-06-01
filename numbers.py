# -*- coding: utf-8 -*-
"""
Created on Tue May 30 14:57:25 2017

@author: shirmanov
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

df = pd.read_csv('train.csv')

labels = df.iloc[:, 0]
imgs = df.iloc[:, 1:]
imgs_for_show = imgs.values.reshape(len(imgs), 28, 28)

train_X, test_X, train_y, test_y = train_test_split(
        imgs, labels, test_size=0.3)

print(train_X.shape, train_y.shape)
    
svm = SVC(kernel='linear', C=1)
svm.fit(train_X, train_y)

print(svm.score(test_X, test_y))

mlp = MLPClassifier(
        solver='lbfgs', hidden_layer_sizes=(100, 100), alpha=1e-5)
mlp.fit(train_X, train_y)

print(mlp.score(test_X, test_y))