from pipeline import image_loader
from pipeline import feature_extraction
from pipeline import classification as clas
from pipeline import clustering as clus
# from pipeline import grid_search
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.model_selection import LeaveOneOut

imlabels = [0] * 150  # temporary


def validation(x_data, y_data, method):
    if method == 'test-train':  # test-train 80:20
        X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2)
    elif method == 'loo': # leave-one-out cross-validation
        LeaveOneOut().get_n_splits(images)

        for train_i, test_i in LeaveOneOut().split(images):
            # print("TRAIN:", train_i, "TEST:", test_i)
            X_train, X_test = images[train_i], images[test_i]
            y_train, y_test = imlabels[train_i], imlabels[test_i]
            # print(X_train, X_test, y_train, y_test)
    else:
        X_train, X_test, y_train, y_test = 0
        print("please input a valid method train-test or loo.")

    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    image_path = "BigCats"
    augment = False

    if augment:
        image_loader.augment_images(image_path)

    # xfeatures, xlabels, yfeatures, ylabels = image_loader.load_images(image_path)

    """
    # full images with train-test, model1=KNN, model2=LR, model3 =NB
    X_train, X_test, y_train, y_test = validation(images, imlabels, 'test-train')
    model1, model2, model3 = clas.classification(X_train, X_test, y_train, y_test) #classify features, input

    # full images with leave-one-out cross validation
    X_train, X_test, y_train, y_test = validation(images, imlabels, 'loo')
    model4, model5, model6 = clas.classification(X_train, X_test, y_train, y_test) #classify features, input
    """
    # TODO: IMPLEMENT THIS
    reduced_data, imlabels = image_loader.load_images(image_path) #extract features from images

    # reduced data with train-test
    X_train, X_test, y_train, y_test = validation(reduced_data, imlabels, 'test-train')
    rmodel1, rmodel2, rmodel3 = clas.classification(X_train, X_test, y_train, y_test) #classify features reduced images

    # reduced data with leave-one-out
    #X_train, X_test, y_train, y_test = validation(reduced_data, imlabels, 'loo')
    #rmodel4, rmodel5, rmodel6 = clas.classification(X_train, X_test, y_train, y_test) #classify features reduced images

    # clustering: analyse outcome results for original and reduced dataset
    # clus.myDBSCAN(images)
    clus.myDBSCAN(reduced_data)
