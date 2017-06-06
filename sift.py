
# Convenience functions to run a grid search over the classiers and over K in KMeans

import sift_classifier as bow
from sklearn.cluster import MiniBatchKMeans
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def cluster_and_split(labeled_img_paths, y, K):
    """Cluster into K clusters, then split into train/test/val"""
    # MiniBatchKMeans annoyingly throws tons of deprecation warnings that fill up the notebook. Ignore them.
    warnings.filterwarnings('ignore')

    X, cluster_model = bow.cluster_features(
        labeled_img_paths,
        cluster_model=MiniBatchKMeans(n_clusters=K)
    )

    warnings.filterwarnings('default')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    return X_train, X_test, y_train, y_test, cluster_model

def run_svm(X_train, X_test, y_train, y_test, scoring,
    c_vals=[1, 5, 10], gamma_vals=[0.1, 0.01, 0.0001, 0.00001]):

    param_grid = [
      {'C': c_vals, 'kernel': ['linear']},
      #{'C': c_vals, 'gamma': gamma_vals, 'kernel': ['rbf']},
     ]

    print('start training svm')

    #svc = GridSearchCV(SVC(), param_grid, n_jobs=-1, scoring=scoring)
    svc = SVC(C=0.9, kernel='linear')
    svc.fit(X_train, y_train)
    print('train score (%s):'%scoring, svc.score(X_train, y_train))
    test_score = svc.score(X_test, y_test)
    print('test score (%s):'%scoring, test_score)

    mlp = MLPClassifier(hidden_layer_sizes=(10, 2))
    mlp.fit(X_train, y_train)
    print('train score (%s):'%scoring, mlp.score(X_train, y_train))
    test_score_mlp = mlp.score(X_test, y_test)
    print('test score (%s):'%scoring, test_score_mlp)
    
    d = pd.DataFrame([["SVM", test_score*100], ["MLP", test_score_mlp*100]],
                 columns=["Classifier", 'Accuracy'])
    
    sns.set_color_codes("muted")
    sns.barplot(x='Accuracy', y='Classifier', data=d, color="b")

    plt.xlabel('Accuracy %')
    plt.title('Classifier Accuracy')
    plt.show()


    #print(svc.best_estimator_)

    return svc, test_score