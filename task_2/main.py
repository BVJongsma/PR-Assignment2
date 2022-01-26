import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from imblearn.under_sampling import RandomUnderSampler
from sklearn.semi_supervised import LabelPropagation

# when running, get the error: numpy.core._exceptions._ArrayMemoryError:
#               Unable to allocate 34.8 GiB for an array with shape (68353, 68353) and data type float64



#
def plot_acc_f1(data1, data2):
    plt.figure()
    plt.subplot(211)
    plt.plot( range(1, 101), data1)
    plt.ylabel('accuracy score for supervised KNN')

    plt.subplot(2, 1, 2)
    plt.plot(range(1, 101), data2)
    plt.xlabel('iteration')
    plt.ylabel('f1-score score for supervised KNN')
    plt.show()
    return

if __name__ == '__main__':
    path = "creditcard.csv"
    data = pd.read_csv(path)
    print("the shape of data is ", data.shape)
    # data = data[0:1000] #subset to see if code works without taking hours

    data_y = data.Class
    data_x = data.iloc[:, range(2, 29)] # The time and amount columns are not included
    print("The shape of x is ", data_x.shape, " and the shape of y is ", data_y.shape)
    rus = RandomUnderSampler(sampling_strategy=0.05)
    data_x, data_y = rus.fit_resample(data_x, data_y)
    print("The new shape of x is ", data_x.shape, " and the new shape of y is ", data_y.shape)


    
    # 1.prepare/split the data
    x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2, stratify=data_y)
    x30_lab, x70_unlab, y30_lab, y70_unlab = train_test_split(x_train, y_train, test_size=0.7,
                                                              stratify=y_train)  # forget y70_unlab
    print("the data has been split.")

    # data_labeled = x30_lab.append(y30_lab)
    data_labeled = x30_lab.copy()
    data_labeled['class'] = y30_lab.values

    print("we start with the baseline model training.")
    # 2.train baseline model, I choose KNN
    acc_score = [0] * 100
    f1score = [0] * 100
    for i in range(100):
        knn = KNeighborsClassifier(n_neighbors=5)  # evaluate later
        knn.fit(x30_lab, y30_lab.values)
        y_pred = knn.predict(x_test)
        acc_score[i] = accuracy_score(y_test, y_pred)
        f1score[i] = f1_score(y_test, y_pred)
        print("knn:", i, "acc score is", acc_score[i])

    plot_acc_f1(acc_score, f1score)

    # 3.train semi-supervised: labeled #
    y70_noise = [-1] * len(y70_unlab)
    acc_score = [0] * 100
    f1score = [0] * 100

    x_mixed = pd.concat([x30_lab, x70_unlab])
    y_mixed = list(y30_lab.values) + y70_noise

    for i in range(100):
        semis = LabelPropagation(kernel='knn').fit(x_mixed, y_mixed)
        y_pred = semis.predict(x_test)
        acc_score[i] = accuracy_score(y_test, y_pred)
        f1score[i] = f1_score(y_test, y_pred)
        print("semi:", i)
    plot_acc_f1(acc_score, f1score)



    # 4.train baseline model complete dataset
    y_semi = semis.transduction_
    acc_score = [0] * 100
    f1score = [0] * 100
    for i in range(100):
        knn_comp = KNeighborsClassifier().fit(x_mixed, y_semi)
        y_pred = knn_comp.predict(x_test)
        acc_score[i] = accuracy_score(y_test, y_pred)
        f1score[i] = f1_score(y_test, y_pred)
        print("knn_comp:", i)
    plot_acc_f1(acc_score, f1score)



