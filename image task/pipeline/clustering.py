from sklearn.cluster import DBSCAN
from sklearn import preprocessing


def myDBSCAN(x_data, y_data):
    le = preprocessing.LabelEncoder()
    labels = le.fit_transform(y_data)
    clustering = DBSCAN(eps=0.6, min_samples=2).fit(x_data)  # eps???
    labels_DBSCAN = le.fit_transform(clustering.labels_) #no idea what this actually does
    return clustering, labels_DBSCAN

#to test out DBSCAN
if __name__ == '__main__':
    x = [[0, 0], [1, 0], [1, 0], [0, 1], [0, 1], [0, 0.5]]
    y = ['a', 'b', 'b', 'c', 'd', 'd', 'd']
    c, lab = myDBSCAN(x, y)
    print("this is c: ", c)
    print("this is lab: ", lab)
