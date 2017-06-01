# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 13:32:09 2017

@author: shirmanov
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

leaf_1 = cv2.imread('Acer_Opalus1.jpg')
leaf_2 = cv2.imread('Acer_Opalus2.jpg')

def show_rgb_img(img):
    return plt.imshow(cv2.cvtColor(img, cv2.CV_32S))

def to_gray(color_img):
    gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    return gray

leaf_1_gray = to_gray(leaf_1)
leaf_2_gray = to_gray(leaf_2)

plt.imshow(leaf_1_gray, cmap='gray')

def gen_sift_features(gray_img):
    sift = cv2.xfeatures2d.SIFT_create()
    kp, desc = sift.detectAndCompute(gray_img, None)
    return kp, desc

def show_sift_features(gray_img, color_img, kp):
    return plt.imshow(
            cv2.drawKeypoints(
                    gray_img, kp, color_img.copy()))
    
leaf_1_kp, leaf_1_desc = gen_sift_features(leaf_1_gray)
leaf_2_kp, leaf_2_desc = gen_sift_features(leaf_2_gray)

print('Here are what our SIFT features look like for the leaf 1 image')
show_sift_features(leaf_1_gray, leaf_1, leaf_1_kp)

bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

matches = bf.match(leaf_1_desc, leaf_2_desc)
matches = sorted(matches, key = lambda x: x.distance)

N_MATCHES = 100

match_img = cv2.drawMatches(
        leaf_1, leaf_1_kp,
        leaf_2, leaf_2_kp,
        matches[:N_MATCHES], leaf_2.copy(), flags=0)

plt.figure(figsize=(12, 6))
plt.imshow(match_img)

def explain_keypoint(kp):
    print('angle\n', kp.angle)
    print('\nclass_id\n', kp.class_id)
    print('\noctave (image scale where feature is strongest)\n', kp.octave)
    print('\npt (x, y)\n', kp.pt)
    print('\nresponse\n', kp.response)
    print('\nsize\n', kp.size)
    
print(
      'this is an example of a single SIFT keypoint:\n* * *',
      explain_keypoint(leaf_1_kp[0]))

plt.imshow(leaf_1_desc[0].reshape(16, 8), interpolation='none')