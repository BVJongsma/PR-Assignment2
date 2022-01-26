import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from imblearn.under_sampling import RandomUnderSampler
from sklearn.semi_supervised import LabelPropagation

n = 100


def plot_acc_f1(data1, data2, title, plot_title):
    plt.figure().suptitle(title)
    plt.subplot(211)
    plt.plot(range(1, n + 1), data1)
    plt.ylabel('accuracy score')

    plt.subplot(2, 1, 2)
    plt.plot(range(1, n + 1), data2)
    plt.xlabel('iteration')
    plt.ylabel('f1-score score')

    plt.show()
    plt.savefig(plot_title)
    return


if __name__ == '__main__':
    path = "creditcard.csv"
    data = pd.read_csv(path)

    data_y = data.Class
    data_x = data.iloc[:, range(2, 29)]  # The time and amount columns are not included
    rus = RandomUnderSampler(sampling_strategy=0.05)
    data_x, data_y = rus.fit_resample(data_x, data_y)

    acc_score_2, f1score_2 = [0.0] * n, [0] * n
    acc_score_3, f1score_3 = [0.0] * n, [0] * n
    acc_score_4, f1score_4 = [0.0] * n, [0] * n

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

    plot_acc_f1(acc_score_2, f1score_2, "step 2: KNN supervised", "step2.png")
    print("The mean of accuracy score for supervised baseline model is ", sum(acc_score_2) / n)
    print("The mean of F1 score for supervised baseline model is ", sum(f1score_2) / n)

    plot_acc_f1(acc_score_3, f1score_3, "step 3: semi-supervised" "step3.png")
    print("The mean of accuracy score for semi-supervised model is ", sum(acc_score_3) / n)
    print("The mean of F1 score for semi-supervised model is ", sum(f1score_3) / n)

    plot_acc_f1(acc_score_4, f1score_4, "step 4: full model", "step4.png")
    print("The mean of accuracy score for semi-supervised model is ", sum(acc_score_4) / n)
    print("The mean of F1 score for semi-supervised model is ", sum(f1score_4) / n)
