from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression  # regularization applied by default
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import VotingClassifier
import numpy as np

#KNN classification
def KNN(X_train, X_test, y_train, y_test):
    knn = KNeighborsClassifier().fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    acc_score = metrics.accuracy_score(y_test, y_pred)
    return knn, acc_score

#logistic classification
def logistic_regression(X_train, X_test, y_train, y_test):
    lr = LogisticRegression().fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    acc_score = metrics.accuracy_score(y_test, y_pred)
    return lr, acc_score

#naive bayes classification
def naive_bayes(X_train, X_test, y_train, y_test):
    gnb = GaussianNB().fit(X_train, y_train)
    y_pred = gnb.predict(X_test)
    acc_score = metrics.accuracy_score(y_test, y_pred)
    return gnb, acc_score

#ensemble classification
def ensemble(X_train, X_test, y_train, y_test):
    # group / ensemble of models
    estimator = []
    estimator.append(('KNN', KNeighborsClassifier(n_neighbors=11, weights='uniform')))
    estimator.append(('LR', LogisticRegression(max_iter=100, penalty='none', solver='newton-cg')))
    estimator.append(('NB', GaussianNB()))

    # Voting Classifier with hard voting
    vot_hard = VotingClassifier(estimators=estimator, voting='hard')
    vot_hard.fit(X_train, y_train)
    y_pred = vot_hard.predict(X_test)

    # using accuracy_score metric to predict accuracy
    hard_score = metrics.accuracy_score(y_test, y_pred)
    # print("Hard Voting Score", hard_score)

    # Voting Classifier with soft voting
    vot_soft = VotingClassifier(estimators=estimator, voting='soft')
    vot_soft.fit(X_train, y_train)
    y_pred = vot_soft.predict(X_test)

    # using accuracy_score
    soft_score = metrics.accuracy_score(y_test, y_pred)
    # print("Soft Voting Score", soft_score)

    return hard_score, soft_score


# classification using all 5 models (incl. 2 ensemble) using train:test 80:20
def classification(X_train, X_test, y_train, y_test):
    knn_model, knn_acc = KNN(X_train, X_test, y_train, y_test)
    lr_model, lr_acc = logistic_regression(X_train, X_test, y_train, y_test)
    nb_model, nb_acc = naive_bayes(X_train, X_test, y_train, y_test)
    hard_score, soft_score = ensemble(X_train, X_test, y_train, y_test)

    return knn_model, knn_acc, lr_model, lr_acc, nb_model, nb_acc, hard_score, soft_score

# classification using all 5 models (incl. 2 ensemble0 using leave one out cross validation
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
