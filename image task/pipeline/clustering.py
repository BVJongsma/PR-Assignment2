from sklearn.cluster import DBSCAN
from sklearn import preprocessing
from sklearn.metrics import silhouette_score


def myDBSCAN(x_data):
    le = preprocessing.LabelEncoder()
    clustering = DBSCAN(eps=0.6, min_samples=2).fit(x_data)  # adjust eps and min_samples if necessary for our actual task
    labels = le.fit_transform(clustering.labels_)

    #evaluation
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    try:
        print("Silhouette Coefficient: %0.3f", silhouette_score(x_data, labels))
    except:
        print("An exception occured.")
    return clustering, labels


# to test out DBSCAN
if __name__ == '__main__':
    x = [[0, 0], [1, 0], [1, 0], [0, 1], [0, 1], [0, 0.5]]
    c, lab = myDBSCAN(x)
    x['labels'] = lab
    print("this is c: ", c)
    print("this is lab: ", lab)
