from sklearn.cluster import DBSCAN
from sklearn import preprocessing
from sklearn.metrics import silhouette_score


# clustering using DBSCAN
def myDBSCAN(x_data):
    le = preprocessing.LabelEncoder()
    clustering = DBSCAN(eps=0.6, min_samples=2).fit(x_data)
    labels = le.fit_transform(clustering.labels_)
    score = 999
    try:
        # print("Silhouette Coefficient: %0.3f", silhouette_score(x_data, labels))
        score = silhouette_score(x_data, labels)
    except:
        pass

    return clustering, labels, score
