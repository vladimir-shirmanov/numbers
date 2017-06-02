# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 17:20:33 2017

@author: shirmanov
"""

import cv2

img_descs = []

img = cv2.imread('images/1.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
sift = cv2.xfeatures2d.SIFT_create()
kp, desc = sift.detectAndCompute(gray, None)
# img_keypoints[img_path] = kp
img_descs.append(desc)

img2 = cv2.imread('images/2.jpg')
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
sift = cv2.xfeatures2d.SIFT_create()
kp, desc = sift.detectAndCompute(gray2, None)
# img_keypoints[img_path] = kp
img_descs.append(desc)