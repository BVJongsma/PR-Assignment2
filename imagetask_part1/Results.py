import pandas as pd

# path = "orboriginalDataAccuracy.csv"
# path = "siftoriginalDataAccuracy.csv"
# path =  "orbreducedDataAccuracy.csv"
path = "siftreducedDataAccuracy.csv"

j=0
mean = [0]*4
meandf = pd.DataFrame(columns=['KNN', 'lr', 'NB', 'DBSCAN'])

if __name__ == '__main__':
    df = pd.read_csv(path)
    for i in range(4):
        row = df.iloc[range(0, 100), i]

        mean[i] = (sum(row) / len(row))
        print(df.columns[i], mean[i])
        j += 1

        """"
    df2 = pd.DataFrame(mean, columns=['KNN', 'lr', 'NB', 'DBSCAN'])
    print(df2)
    meandf.append(df2)
    print(meandf)
    meandf.to_csv('means.csv')
        """