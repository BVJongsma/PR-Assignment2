from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression  # regularization applied by default
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
import numpy as np

# https://www.datacamp.com/community/tutorials/k-nearest-neighbor-classification-scikit-learn
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
# https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors
# .KNeighborsClassifier


def KNN(X_train, X_test, y_train, y_test):
    knn = KNeighborsClassifier().fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    acc_score = metrics.accuracy_score(y_test,
                                       y_pred)  # used accuracy score here as evaluation method just for the sake of it
    print("Accuracy score for KNN: ", acc_score)
    return knn, acc_score


def logistic_regression(X_train, X_test, y_train, y_test):
    lr = LogisticRegression().fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    acc_score = metrics.accuracy_score(y_test, y_pred)  # used accuracy score here as evaluation for the hell of it
    print("Accuracy score for Logistic Regression: ", acc_score)
    return lr, acc_score


def naive_bayes(X_train, X_test, y_train, y_test):
    gnb = GaussianNB().fit(X_train, y_train)
    y_pred = gnb.predict(X_test)
    acc_score = metrics.accuracy_score(y_test, y_pred)
    print("Accuracy score for Naive Bayes: ", acc_score)
    return gnb, acc_score


def classification(X_train, X_test, y_train, y_test):
    knn_model, knn_acc = KNN(X_train, X_test, y_train, y_test)
    lr_model, lr_acc = logistic_regression(X_train, X_test, y_train, y_test)
    nb_model, nb_acc = naive_bayes(X_train, X_test, y_train, y_test)

    return knn_model, knn_acc, lr_model, lr_acc, nb_model, nb_acc


def classificationloo(data, labels):
    cv = LeaveOneOut()

    knn_model = KNeighborsClassifier()
    knn_scores = cross_val_score(knn_model, data, labels, scoring='neg_mean_absolute_error',
                         cv=cv, n_jobs=-1)
    knn_acc = np.mean(np.absolute(knn_scores))

    lr_model = LogisticRegression()
    lr_scores = cross_val_score(lr_model, data, labels, scoring='neg_mean_absolute_error',
                                 cv=cv, n_jobs=-1)
    lr_acc = np.mean(np.absolute(lr_scores))

    nb_model = GaussianNB()
    nb_scores = cross_val_score(nb_model, data, labels, scoring='neg_mean_absolute_error',
                                 cv=cv, n_jobs=-1)
    nb_acc = np.mean(np.absolute(nb_scores))

    return knn_model, knn_acc, lr_model, lr_acc, nb_model, nb_acc
