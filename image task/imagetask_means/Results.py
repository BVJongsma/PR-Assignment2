""" Code in this file is used to create plots/visuals/tables for the report """
import pandas as pd

path = "orboriginalDataAccuracy.csv"
# path = "siftoriginalDataAccuracy.csv"
# path =  "orbreducedDataAccuracy.csv"
# path = "siftreducedDataAccuracy.csv"
# path = "orbaugmentedoriginalDataAccuracy.csv"
# path = "siftaugmentedoriginalDataAccuracy.csv"
# path = "orbaugmentedreducedDataAccuracy.csv"
# path = "siftaugmentedreducedDataAccuracy.csv"

mean = [0] * 6

if __name__ == '__main__':
    df = pd.read_csv(path)

    # calculate and print the mean for every column in the csv file.
    for i in range(6):
        n = len(df.iloc[:, i])
        row = df.iloc[range(0, n), i]

        mean[i] = (sum(row) / len(row)) * n
        if df.columns[i] == 'DBSCAN':
            mean[i] /= n
            print('DBSCAN %', (mean[i] + 1) * 50)
        print(df.columns[i], mean[i])
