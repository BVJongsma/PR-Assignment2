from sklearn.cluster import DBSCAN
from sklearn import preprocessing
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples

import pandas as pd


def myDBSCAN(x_data):
    le = preprocessing.LabelEncoder()
    labels = le.fit_transform(y_data)
    clustering = DBSCAN(eps=0.6, min_samples=2).fit(x_data)  # eps???
    labels_DBSCAN = le.fit_transform(clustering.labels_) #no idea what this actually does
    return clustering, labels_DBSCAN

#to test out DBSCAN
if __name__ == '__main__':
    x = [[0, 0], [1, 0], [1, 0], [0, 1], [0, 1], [0, 0.5]]
    #xdf = pd.DataFrame(x)
    #y = ['a', 'b', 'b', 'c', 'd', 'd', 'd']
    c, lab = myDBSCAN(x)
    x['labels'] = lab
    print("this is c: ", c)
    print("this is lab: ", lab)
