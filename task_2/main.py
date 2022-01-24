import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.semi_supervised import LabelPropagation

# when running, get the error: numpy.core._exceptions._ArrayMemoryError:
#               Unable to allocate 34.8 GiB for an array with shape (68353, 68353) and data type float64


if __name__ == '__main__':
    path = "..\\creditcard.csv"
    data = pd.read_csv(path)
    data_y = data.Class
    data_x = data.iloc[:, range(2, 29)]  # The time and amount columns are not included

    # 1.prepare/split the data
    x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2, stratify=data_y)
    x30_lab, x70_unlab, y30_lab, y70_unlab = train_test_split(x_train, y_train, test_size=0.7,
                                                              stratify=y_train)  # forget y70_unlab
    data_labeled = x30_lab.append(y30_lab)
    print(data_labeled)

    # 2.train baseline model, I choose KNN
    knn = KNeighborsClassifier(n_neighbors=5)  # evaluate later
    knn.fit(x30_lab, y30_lab)
    y_pred = knn.predict(x_test)
    print("accuracy score for step 2: ", accuracy_score(y30_lab, y_pred))
    print("f1 score for step 2: ", f1_score(y30_lab, y_pred))

    # 3.train semi-supervised: labeled #
    # TODO: figure out what is meant by step 3 exactly. I think you only train one model it says to
    #      train on both unlabeled and labeled sets.
    y70_noise = [-1] * len(y70_unlab)
    semis = LabelPropagation().fit(x30_lab, y30_lab)
    y_pred = semis.predict(x_test)
    print("accuracy score for labeled step 3: ", accuracy_score(y30_lab, y_pred))
    print("f1 score for labeled step 3: ", f1_score(y30_lab, y_pred))

    # 3.train semi-supervised: unlabeled
    semis = LabelPropagation().fit(x70_unlab, y70_unlab)
    y_pred = semis.predict(x_test)
    print("accuracy score for labeled step 3: ", accuracy_score(y70_noise, y_pred)) #TODO: what to fill in for y_true? y_lab?
    print("f1 score for labeled step 3: ", f1_score(y70_noise, y_pred))

    # 4.train baseline model complete dataset
    y_semi = semis.transfuction_
    y_train = y30_lab.append(y_semi) # TODO: is this correct?
    knn = KNeighborsClassifier().fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    print("accuracy score for step 4: ", accuracy_score(y30_lab, y_pred))
    print("f1 score for step 4: ", f1_score(y30_lab, y_pred))

