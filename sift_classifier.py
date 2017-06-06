# Ian London 2016
# tools to run a Visual Bag of Words classifier on any images

import cv2
import numpy as np
import pandas as pd

def read_image(path):
    img = cv2.imread(path)
    if img is None:
        raise IOError("Unable to open '%s'. Are you sure it's a valid image path?")
    return img

def gen_labeled_img_paths():
    df = pd.read_csv('train.csv')
    labeled_keys = dict(zip(df['species'].unique(), range(len(df['species'].unique()))))
    df['species_id'] = df['species'].replace(labeled_keys)
    return [['images/{}.jpg'.format(path), label] 
             for path, label 
             in zip(df['id'], df['species_id'])]


def gen_sift_features(labeled_img_paths):
    """
    Generate SIFT features for images
    Parameters:
    -----------
    labeled_img_paths : list of lists
        Of the form [[image_path, label], ...]
    Returns:
    --------
    img_descs : list of SIFT descriptors with same indicies as labeled_img_paths
    y : list of corresponding labels
    """
    # img_keypoints = {}
    img_descs = []

    print('generating SIFT descriptors for %i images' % len(labeled_img_paths))

    for img_path, label in labeled_img_paths:
        img = read_image(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sift = cv2.xfeatures2d.SIFT_create()
        kp, desc = sift.detectAndCompute(gray, None)
        # img_keypoints[img_path] = kp
        img_descs.append(desc)

    print('SIFT descriptors generated.')
    
    print(len(img_descs))

    y = np.array(labeled_img_paths)[:,1]

    return img_descs, y


def cluster_features(labeled_img_paths, cluster_model):
    """
    Cluster the training features using the cluster_model
    and convert each set of descriptors in img_descs
    to a Visual Bag of Words histogram.
    Parameters:
    -----------
    labeled_img_paths : list of lists
        Of the form [[image_path, label], ...]
    cluster_model : clustering model (eg KMeans from scikit-learn)
        The model used to cluster the SIFT features
    Returns:
    --------
    X, cluster_model :
        X has K feature columns, each column corresponding to a visual word
        cluster_model has been fit to the training set
    """
    n_clusters = cluster_model.n_clusters

    # # Generate the SIFT descriptor features
    img_descs, y = gen_sift_features(labeled_img_paths)
    # # Generate indexes of training rows
    total_rows = len(img_descs)

    # Concatenate all descriptors in the training set together
    training_descs = [img_descs[i] for i in range(total_rows)]
    
    print(len(img_descs))
    
    all_train_descriptors = [desc for desc_list in training_descs for desc in desc_list]
    all_train_descriptors = np.array(all_train_descriptors)
    
    if all_train_descriptors.shape[1] != 128:
        raise ValueError('Expected SIFT descriptors to have 128 features, got', all_train_descriptors.shape[1])

    print('{0} descriptors before clustering'.format(all_train_descriptors.shape[0]))

    # Cluster descriptors to get codebook
    print('Using clustering model {}...'.format(repr(cluster_model)))
    print('Clustering on training set to get codebook of {0} words'.format(n_clusters))


    # train kmeans or other cluster model on those descriptors selected above
    cluster_model.fit(all_train_descriptors)
    print('done clustering. Using clustering model to generate BoW histograms for each image.')

    # compute set of cluster-reduced words for each image
    img_clustered_words = [cluster_model.predict(raw_words) for raw_words in img_descs]

    # finally make a histogram of clustered word counts for each image. These are the final features.
    img_bow_hist = np.array(
        [np.bincount(clustered_words, minlength=n_clusters) for clustered_words in img_clustered_words])

    X = img_bow_hist
    print('done generating BoW histograms.')

    return X, cluster_model


def img_to_vect(img_path, cluster_model):
    """
    Given an image path and a trained clustering model (eg KMeans),
    generates a feature vector representing that image.
    Useful for processing new images for a classifier prediction.
    """

    img = read_image(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kp, desc = sift.detectAndCompute(gray, None)

    clustered_desc = cluster_model.predict(desc)
    img_bow_hist = np.bincount(clustered_desc, minlength=cluster_model.n_clusters)

    # reshape to an array containing 1 array: array[[1,2,3]]
    # to make sklearn happy (it doesn't like 1d arrays as data!)
    return img_bow_hist.reshape(1,-1)
