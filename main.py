# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 15:25:03 2017

@author: shirmanov
"""

import sift_classifier as sc
import sift as s

labeled_img_paths = sc.gen_labeled_img_paths()
y = [x[1] for x in labeled_img_paths]

X_train, X_test, y_train, y_test, cluster_model = s.cluster_and_split(labeled_img_paths, y, 500)

svm, test_score = s.run_svm(X_train, X_test, y_train, y_test, None)