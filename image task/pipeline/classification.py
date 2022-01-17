import pandas as pd
import matplotlib.pyplot as plt
import image_loader as il
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression #regularization applied by default
from sklearn import metrics
import numpy as np
# https://www.datacamp.com/community/tutorials/k-nearest-neighbor-classification-scikit-learn
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
#https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier

""" Code for loading/augmenting images """
"""
dataset = il.load_images("../BigCats")
feature_names = ['one', 'two']

"""

X_train, X_test, y_train, y_test = train_test_split(images, reduced_data) #doesn't run since images& reduced data aren't in this file


def KNN(data):
    knn = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    acc_score = metrics.accuracy_score(y_test, y_pred) #used accuracy score here as evaluation method just for the sake of it
    print("accuracy score: ", acc_score)


def logistic_regression(data):
    lr = LogisticRegression().fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    acc_score = metrics.accuracy_score(y_test, y_pred) #used accuracy score here as evaluation for the hell of it
    print("logistic regression")

def third_classification():

""""
if __name__ == '__main__':

    num = int(input("Enter number. 1. KNN 2.Logistic Regression:"))
    if num == 1:
        KNN(dataset)
    elif num == 2:
        logistic_regression(dataset)
    else:
        print("please input a valid number. you've chosen ", num)
        print("the type of num is ", type(num), " and the type of 1 is ", type(1))
"""