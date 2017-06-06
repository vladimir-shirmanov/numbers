# -*- coding: utf-8 -*-
"""
Created on Tue Jun 06 06:12:56 2017

@author: vladi
"""

import numpy as np
import scipy as sp
import scipy.ndimage as ndi
import pandas as pd
from skimage import measure
from sklearn import metrics
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt


# ----------------------------------------------------- I/O ---

def read_img(img_no):
    """reads image from disk"""
    return mpimg.imread('images/' + str(img_no) + '.jpg')

def get_imgs_by_ids(ids):
    """convenience function, yields random sample from leaves"""
    for img_no in ids:
        yield img_no, preprocess(read_img(img_no))


def get_imgs(num):
    """convenience function, yields random sample from leaves"""
    for img_no in range(1, num + 1):
        yield img_no, preprocess(read_img(img_no))


# -------------------- preprocessing ---

def threshold(img, threshold=250):
    """splits img to 0 and 255 values at threshold"""
    return ((img > threshold) * 255).astype(img.dtype)


def portrait(img):
    """makes all leaves stand straight"""
    y, x = np.shape(img)
    return img.transpose() if x > y else img


def resample(img, size):
    """resamples img to size without distorsion"""
    ratio = float(size) / max(np.shape(img))
    return sp.misc.imresize(img, ratio, mode='L', interp='nearest')


def fill(img, size=500, tolerance=0.95):
    """extends the image if it is signifficantly smaller than size"""
    y, x = np.shape(img)

    if x <= size:
        pad = np.zeros((y, int((size - x))), dtype=int)
        img = np.concatenate((pad, img), axis=1)

    if y <= size:
        pad = np.zeros((int((size - y)), size), dtype=int)
        img = np.concatenate((pad, img), axis=0)

    return img


# ----------------------------------------------------- postprocessing ---

def standardize(arr1d):
    """move mean to zero, 1st SD to -1/+1"""
    return (arr1d - arr1d.mean()) / arr1d.std()


def coords_to_cols(coords):
    """from x,y pairs to feature columns"""
    return coords[::, 1], coords[::, 0]


def get_contour(img):
    """returns the coords of the longest contour"""
    return max(measure.find_contours(img, .8), key=len)


def downsample_contour(coords, bins=512):
    """splits the array to ~equal bins, and returns one point per bin"""
    edges = np.linspace(0, coords.shape[0],
                        num=bins).astype(int)
    for b in range(bins - 1):
        yield [coords[edges[b]:edges[b + 1], 0].mean(),
               coords[edges[b]:edges[b + 1], 1].mean()]


def get_center(img):
    """so that I do not have to remember the function ;)"""
    return ndi.measurements.center_of_mass(img)


# ----------------------------------------------------- feature engineering ---

def extract_shape(img):
    """
    Expects prepared image, returns leaf shape in img format.
    The strength of smoothing had to be dynamically set
    in order to get consistent results for different sizes.
    """
    size = int(np.count_nonzero(img) / 1000)
    brush = int(5 * size / size ** .75)
    return ndi.gaussian_filter(img, sigma=brush, mode='nearest') > 200


def near0_ix(timeseries_1d, radius=5):
    """finds near-zero values in time-series"""
    return np.where(timeseries_1d < radius)[0]


def dist_line_line(src_arr, tgt_arr):
    """
    returns 2 tgt_arr length arrays, 
    1st is distances, 2nd is src_arr indices
    """
    return np.array(sp.spatial.cKDTree(src_arr).query(tgt_arr))


def dist_line_point(src_arr, point):
    """returns 1d array with distances from point"""
    point1d = [[point[0], point[1]]] * len(src_arr)
    return metrics.pairwise.paired_distances(src_arr, point1d)


def index_diff(kdt_output_1):
    """
    Shows pairwise distance between all n and n+1 elements.
    Useful to see, how the dist_line_line maps the two lines.
    """
    return np.diff(kdt_output_1)


# ----------------------------------------------------- wrapping functions ---

# wrapper function for all preprocessing tasks    
def preprocess(img, do_portrait=True, do_resample=500,
               do_fill=True, do_threshold=250):
    """ prepares image for processing"""
    if do_portrait:
        img = portrait(img)
    if do_resample:
        img = resample(img, size=do_resample)
    if do_fill:
        img = fill(img, size=do_resample)
    if do_threshold:
        img = threshold(img, threshold=do_threshold)

    return img


# wrapper function for feature extraction tasks
def get_std_contours(img):
    """from image to standard-length countour pairs"""

    # shape in boolean n:m format
    blur = extract_shape(img)

    # contours in [[x,y], ...] format
    blade = np.array(list(downsample_contour(get_contour(img))))
    shape = np.array(list(downsample_contour(get_contour(blur))))

    # flagging blade points that fall inside the shape contour
    # notice that we are loosing subpixel information here
    blade_y, blade_x = coords_to_cols(blade)
    blade_inv_ix = blur[blade_x.astype(int), blade_y.astype(int)]

    # img and shape centers
    shape_cy, shape_cx = get_center(blur)
    blade_cy, blade_cx = get_center(img)

    # img distance, shape distance (for time series plotting)
    blade_dist = dist_line_line(shape, blade)
    shape_dist = dist_line_point(shape, [shape_cx, shape_cy])

    # fixing false + signs in the blade time series
    blade_dist[0, blade_inv_ix] = blade_dist[0, blade_inv_ix] * -1

    return {'shape_img': blur,
            'shape_contour': shape,
            'shape_center': (shape_cx, shape_cy),
            'shape_series': [shape_dist, range(len(shape_dist))],
            'blade_img': img,
            'blade_contour': blade,
            'blade_center': (blade_cx, blade_cy),
            'blade_series': blade_dist,
            'inversion_ix': blade_inv_ix}

def get_prepared_features(data_frame):
    keys = dict(zip(data_frame['species'].unique(), range(len(data_frame['species'].unique()))))
    y = data_frame['species'].replace(keys).tolist()
    data_len = len(data_frame)
    imgs = list(get_imgs(data_len))
    features = get_std_contours(imgs[0][1])
    shape = features['shape_series'][0]
    blade = features['blade_series'][0]
    X = np.append(blade, shape)

    for title, img in imgs[1:]:
        features = get_std_contours(img)
        shape = features['shape_series'][0]
        blade = features['blade_series'][0]
        series = np.append(blade, shape)
        X = np.vstack((X, series))
    return X, y

df = pd.read_csv('train.csv')
keys = dict(zip(df['species'].unique(), range(len(df['species'].unique()))))
y = df['species'].replace(keys).tolist()
x_ids = df['id'].values
          
X = np.array([img.flatten() for img_no, img in get_imgs_by_ids(x_ids)])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

"""
svm = svm.SVC(C=1)
svm.fit(X_train, y_train)
print('train scores {}'.format(svm.score(X_train, y_train)))
print('test scores {}'.format(svm.score(X_test, y_test)))
"""

mlp = MLPClassifier(hidden_layer_sizes=(5,2))
mlp.fit(X_train, y_train)
print('train scores {}'.format(mlp.score(X_train, y_train)))
print('test scores {}'.format(mlp.score(X_test, y_test)))
