# -*- coding: utf-8 -*-
"""
Created on Tue May 30 14:57:25 2017

@author: shirmanov
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('train.csv')

labels = df.iloc[:, 0]
imgs = df.iloc[:, 1:]
imgs_for_show = imgs.values.reshape(len(imgs), 28, 28)

plt.figure(1)
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(imgs_for_show[i])