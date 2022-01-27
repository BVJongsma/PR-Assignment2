import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from imblearn.under_sampling import RandomUnderSampler
from sklearn.semi_supervised import LabelPropagation

# the amount of iterations
n = 100


# plot the distribution of the different steps
def plot_boxplot(data1, data2, data3, title):
    data = [data3, data2, data1]
    fig = plt.figure().suptitle(title)
    plt.boxplot(data, vert=False)
    plt.yticks([1, 2, 3], ['step 4', 'step 3', 'step 2'])
    plt.show()
    return


if __name__ == '__main__':
    # load the data & initialise
    path = "creditcard.csv"
    data = pd.read_csv(path)

    acc_score_2, f1score_2 = [0.0] * n, [0] * n
    acc_score_3, f1score_3 = [0.0] * n, [0] * n
    acc_score_4, f1score_4 = [0.0] * n, [0] * n

    #undersampling
    data_y = data.Class
    data_x = data.iloc[:, range(2, 29)]  # The time and amount columns are not included
    rus = RandomUnderSampler(sampling_strategy=0.05)
    data_x, data_y = rus.fit_resample(data_x, data_y)

    #for n iterations, do all 4 steps
    for i in range(n):
        # 1.prepare/split the data
        x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2, stratify=data_y)
        x30_lab, x70_unlab, y30_lab, y70_unlab = train_test_split(x_train, y_train, test_size=0.7,
                                                                  stratify=y_train)  # forget y70_unlab

        data_labeled = x30_lab.copy()
        data_labeled['class'] = y30_lab.values

        # 2.train baseline model, I choose KNN
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(x30_lab, y30_lab.values)
        y_pred = knn.predict(x_test)
        acc_score_2[i] = accuracy_score(y_test, y_pred)
        f1score_2[i] = f1_score(y_test, y_pred)

        # 3.train semi-supervised: labeled #
        y70_noise = [-1] * len(y70_unlab)
        x_mixed = pd.concat([x30_lab, x70_unlab])
        y_mixed = list(y30_lab.values) + y70_noise

        semis = LabelPropagation(kernel='knn').fit(x_mixed, y_mixed)
        y_pred = semis.predict(x_test)
        acc_score_3[i] = accuracy_score(y_test, y_pred)
        f1score_3[i] = f1_score(y_test, y_pred)

        # 4.train baseline model complete dataset
        y_semi = semis.transduction_
        knn_comp = KNeighborsClassifier().fit(x_mixed, y_semi)
        y_pred = knn_comp.predict(x_test)
        acc_score_4[i] = accuracy_score(y_test, y_pred)
        f1score_4[i] = f1_score(y_test, y_pred)

    #print the mean scores over all n iterations
    print("The mean of accuracy score for supervised baseline model is ", sum(acc_score_2) / n)
    print("The mean of F1 score for supervised baseline model is ", sum(f1score_2) / n)

    print("The mean of accuracy score for semi-supervised model is ", sum(acc_score_3) / n)
    print("The mean of F1 score for semi-supervised model is ", sum(f1score_3) / n)

    print("The mean of accuracy score for semi-supervised model is ", sum(acc_score_4) / n)
    print("The mean of F1 score for semi-supervised model is ", sum(f1score_4) / n)

    #plot the boxplot of F1-score and Acc-score
    plot_boxplot(acc_score_2, acc_score_3, acc_score_4, 'Accuracy score for the different steps')
    plot_boxplot(f1score_2, f1score_3, f1score_4, 'F1-score for the different steps')
