from pipeline import image_loader
from pipeline import classification as clas
from pipeline import clustering as clus
from pipeline import grid_search as gs

from sklearn.model_selection import train_test_split
from csv import writer


# split up the data into train-test splits
def validation(x_data, y_data, method):
    if method == 'test-train':
        X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2)
    else:
        X_train, X_test, y_train, y_test = 0
        print("please input a valid method train-test or loo.")

    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    image_path = "BigCats"
    augment = True
    extraction_method = 'sift'              # should be either sift or orb
    feature_optimization = False            # Optimization of feature extraction will be run if this is true
    grid_search = False                     # Grid search step will be run if this is true
    classification_and_clustering = True    # Classification and clustering will be run if this is true
    range_limit = 40                        # Determines how often classification and clustering will be run
                                            # Note that if augment is True, range should not be too otherwise
                                            # The program will take a very long time to execute

    if augment:
        image_loader.augment_images(image_path)
        image_path = "BigCatsAugmented"

    if feature_optimization:
        for features in [50, 100, 150, 200, 250, 500]:
            print("Testing with features:", features)
            with open('csv/' + str(features) + extraction_method + 'accuracy.csv', 'w+', newline='') as file:
                csv_writer = writer(file, delimiter=',')
                csv_writer.writerow(['KNN', 'lr', 'NB', 'DBSCAN'])
                for i in range(0, range_limit):
                    print(i + 1, "of", range_limit)
                    reduced_data, imlabels = image_loader.load_images(image_path, extraction_method, features)  # extract features from images using ORB

                    # reduced data with train-test
                    X_train, X_test, y_train, y_test = validation(reduced_data, imlabels, 'test-train')
                    rmodel1, racc_1, rmodel2, racc_2, rmodel3, racc_3 = clas.classification(X_train, X_test, y_train,
                                                                                            y_test)  # classify features reduced images
                    _, _, score = clus.myDBSCAN(reduced_data)

                    csv_writer.writerow([racc_1,racc_2,racc_3,score])
                file.close()
    if grid_search:
        reduced_data, imlabels = image_loader.load_images(image_path, extraction_method)  # extract features from images
        rmodel1, racc_1, rmodel2, racc_2 = gs.grid_search(reduced_data, imlabels)  # classify features reduced images
    if classification_and_clustering:
        if augment:
            augmentPath = 'augmented'
        else:
            augmentPath = ''
        print("loading features")
        data, imlabels = image_loader.load_images(image_path, extraction_method)  # extract features from images

        print("loading reduced features")
        reduced_data, reduced_imlabels = image_loader.load_images(image_path, extraction_method,
                                                          100)  # extract features from images using ORB
        # reduced data train/test
        with open('csv/' + extraction_method + augmentPath + 'reducedDataAccuracy.csv', 'w+', newline='') as file:
            csv_writer = writer(file, delimiter=',')
            csv_writer.writerow(['KNN', 'lr', 'NB', 'hard-vote', 'soft-vote', 'DBSCAN'])
            for i in range(0, range_limit):
                print(i + 1, "of", range_limit)

                X_train, X_test, y_train, y_test = validation(reduced_data, reduced_imlabels, 'test-train')
                rmodel1, racc_1, rmodel2, racc_2, rmodel3, racc_3, hard_score, soft_score = clas.classification(X_train, X_test, y_train,
                                                                                        y_test)
                _, _, score = clus.myDBSCAN(reduced_data)

                csv_writer.writerow([racc_1, racc_2, racc_3, hard_score, soft_score, score])
            file.close()
        """
        # reduced data with LOO
        with open('csv/' + extraction_method + augmentPath + 'reducedDataLOO.csv', 'w+', newline='') as file:
            csv_writer = writer(file, delimiter=',')
            csv_writer.writerow(['KNN', 'lr', 'NB'])
            for i in range(0, range_limit):
                print(i + 1, "of", range_limit)

                rmodel1, racc_1, rmodel2, racc_2, rmodel3, racc_3 = clas.classificationloo(reduced_data, reduced_imlabels)  # classify features reduced images

                csv_writer.writerow([racc_1, racc_2, racc_3])
            file.close()
        """
        # original data train/test
        with open('csv/' + extraction_method + augmentPath + 'originalDataAccuracy.csv', 'w+', newline='') as file:
            csv_writer = writer(file, delimiter=',')
            csv_writer.writerow(['KNN', 'lr', 'NB', 'hard-vote', 'soft-vote', 'DBSCAN'])
            for i in range(0, range_limit):
                print(i + 1, "of", range_limit)

                X_train, X_test, y_train, y_test = validation(data, imlabels, 'test-train')
                rmodel1, racc_1, rmodel2, racc_2, rmodel3, racc_3, hard_score, soft_score = clas.classification(X_train, X_test,
                                                                                        y_train,
                                                                                        y_test)
                _, _, score = clus.myDBSCAN(data)
                csv_writer.writerow([racc_1, racc_2, racc_3, hard_score, soft_score, score])
            file.close()
        """
        # original data with LOO
        with open('csv/' + extraction_method + augmentPath + 'originalDataLOO.csv', 'w+', newline='') as file:
            csv_writer = writer(file, delimiter=',')
            csv_writer.writerow(['KNN', 'lr', 'NB'])
            for i in range(0, range_limit):
                print(i + 1, "of", range_limit)

                rmodel1, racc_1, rmodel2, racc_2, rmodel3, racc_3 = clas.classificationloo(data, imlabels)
                csv_writer.writerow([racc_1, racc_2, racc_3])
            file.close()
        """