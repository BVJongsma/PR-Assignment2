from pipeline import image_loader
from pipeline import classification as clas
from pipeline import clustering as clus
from pipeline import grid_search as gs

from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneOut
from csv import writer

def validation(x_data, y_data, method):
    if method == 'test-train':  # test-train 80:20
        X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2)
    elif method == 'loo':  # leave-one-out cross-validation
        loo = LeaveOneOut()
        loo.get_n_splits(x_data)

        for train_i, test_i in loo.split(x_data):
            # print("TRAIN:", train_i, "TEST:", test_i)
            X_train, X_test = x_data[train_i], x_data[test_i]
            y_train, y_test = y_data[train_i], y_data[test_i]
            # print(X_train, X_test, y_train, y_test)
    else:
        X_train, X_test, y_train, y_test = 0
        print("please input a valid method train-test or loo.")

    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    image_path = "BigCats"
    augment = False
    optimization = False
    extraction_method = 'orb'
    feature_optimization = False
    grid_search = False

    if augment:
        image_loader.augment_images(image_path)
        image_path = "BigCatsAugmented"

    if optimization:
        with open('csv/' + extraction_method + 'accuracy.csv', 'w+', newline='') as file:
            csv_writer = writer(file, delimiter=',')
            csv_writer.writerow(['KNN', 'lr', 'NB', 'DBSCAN'])
            for i in range(0, 10):
                reduced_data, imlabels = image_loader.load_images(image_path, extraction_method)  # extract features from images using ORB

                # reduced data with train-test
                X_train, X_test, y_train, y_test = validation(reduced_data, imlabels, 'test-train')
                rmodel1, racc_1, rmodel2, racc_2, rmodel3, racc_3 = clas.classification(X_train, X_test, y_train,
                                                                                        y_test)  # classify features reduced images
                _, _, score = clus.myDBSCAN(reduced_data)

                csv_writer.writerow([racc_1, racc_2, racc_3, score])
            file.close()
    elif feature_optimization:
        range_limit = 10
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
    elif grid_search:
        reduced_data, imlabels = image_loader.load_images(image_path, extraction_method)  # extract features from images
        rmodel1, racc_1, rmodel2, racc_2 = gs.grid_search(reduced_data, imlabels)  # classify features reduced images
    else:
        """
        # full images with train-test, model1=KNN, model2=LR, model3 =NB
        X_train, X_test, y_train, y_test = validation(images, imlabels, 'test-train')
        model1, acc_1, model2, acc_2, model3, acc_3 = clas.classification(X_train, X_test, y_train, y_test) #classify features, input
    
        # full images with leave-one-out cross validation
        X_train, X_test, y_train, y_test = validation(images, imlabels, 'loo')
        model4, acc_4, model5, acc_5, model6, acc_6 = clas.classification(X_train, X_test, y_train, y_test) #classify features, input
        """

        reduced_data, imlabels = image_loader.load_images(image_path, extraction_method)  # extract features from images

        # reduced data with train-test
        X_train, X_test, y_train, y_test = validation(reduced_data, imlabels, 'test-train')
        rmodel1, racc_1, rmodel2, racc_2, rmodel3, racc_3 = clas.classification(X_train, X_test, y_train,
                                                        y_test)  # classify features reduced images

        # reduced data with leave-one-out
        # X_train, X_test, y_train, y_test = validation(reduced_data, imlabels, 'loo')
        rmodel4, racc_4, rmodel5, racc_5, rmodel6, racc_6 = clas.classificationloo(reduced_data, imlabels)  # classify features reduced images

        # clustering: analyse outcome results for original and reduced dataset
        # clus.myDBSCAN(images)
        _, _, score = clus.myDBSCAN(reduced_data)
