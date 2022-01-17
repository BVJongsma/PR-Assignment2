from dataloader import *
import preprocessing
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE

#load the gene data
handler = Dataloader()

#scale all variables to have a mean of 0 and standard deviation of 1
data = preprocessing.scale(handler.data)

#with pca extract eigen pairs that explains 95% of the variance in the data.
pca = PCA(n_components=0.95)

#simultanously calculate eigen pairs and transform our data into the new coordinate frame
principalComponents = pca.fit_transform(data)

#check the amount of dimensions left after pca
print(principalComponents.shape)

#creating labelEncoder
le = preprocessing.LabelEncoder()
labels = le.fit_transform(handler.labels)

#plot 3 most important principle components (3D plot)
color = np.array(['r', 'g', 'b', 'c', 'm'])
fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(projection='3d')
ax.scatter(principalComponents.T[0], principalComponents.T[1], principalComponents.T[2], color=color[labels])
plt.show()

#X_embedded = TSNE(n_components=2, learning_rate='auto',init='random').fit_transform(data)

#split in test and train
X_train, X_test, y_train, y_test = train_test_split(principalComponents, labels, test_size=0.20)

#knn results
model = KNeighborsClassifier(n_neighbors=10)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("KNN Accuracy over the dimension reduced data: ", acc)


#kmeans
kmeans = KMeans(n_clusters=len(np.unique(handler.labels)), random_state=0).fit(principalComponents)
labels_kmeans = le.fit_transform(kmeans.labels_)
