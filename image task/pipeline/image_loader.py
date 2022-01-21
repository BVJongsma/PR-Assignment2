""" Code for loading/augmenting images """
import cv2
import os
import matplotlib.pyplot as plt
import random
import pylab as pl
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
from scipy.cluster.vq import kmeans, vq
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

# example of horizontal shift image augmentation
from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator


def img_list(path):
    return (os.path.join(path, f) for f in os.listdir(path))


# https://machinelearningknowledge.ai/image-classification-using-bag-of-visual-words-model/
def load_images(train_path):
    # randomly split up the data in train/test sets by 80/20 split
    class_names = os.listdir(train_path)
    image_paths = []
    image_classes = []
    i = 0

    for training_name in class_names:
        dir_ = os.path.join(train_path, training_name)
        class_path = img_list(dir_)
        image_paths += class_path
        image_classes += [i] * len(os.listdir(dir_))
        i += 1

    D = []
    for i in range(len(image_paths)):
        D.append((image_paths[i], image_classes[i]))
    dataset = D
    random.shuffle(dataset)
    split_val = round((len(image_paths)) * 0.8)
    train = dataset[:split_val]
    test = dataset[split_val:]

    image_paths, y_train = zip(*train)
    image_paths_test, y_test = zip(*test)

    trainfeatures, testfeatures = extraction_orb(image_paths, y_train, image_paths_test, y_test)

    return np.vstack([trainfeatures, testfeatures]), list(y_train) + list(y_test)


# feature extraction using orb
def extraction_orb(image_paths, y_train, image_paths_test, y_test):
    des_list = []

    orb = cv2.ORB_create()

    for image_path in image_paths:
        im = cv2.imread(image_path)
        kp = orb.detect(im, None)
        keypoints, descriptor = orb.compute(im, kp)
        des_list.append((image_path, descriptor))
    descriptors = des_list[0][1]

    for image_path, descriptor in des_list[1:]:
        descriptors = np.vstack((descriptors, descriptor))

    descriptors_float = descriptors.astype(float)

    k = 200
    voc, variance = kmeans(descriptors_float, k, 1)
    im_features = np.zeros((len(image_paths), k), "float32")

    for i in range(len(image_paths)):
        words, distance = vq(des_list[i][1], voc)
        for w in words:
            im_features[i][w] += 1

    stdslr = StandardScaler().fit(im_features)
    im_features = stdslr.transform(im_features)

    des_list_test = []

    for image_path in image_paths_test:
        image = cv2.imread(image_path)
        kp = orb.detect(image, None)
        keypoints_test, descriptor_test = orb.compute(image, kp)
        des_list_test.append((image_path, descriptor_test))

    test_features = np.zeros((len(image_paths_test), k), "float32")
    for i in range(len(image_paths_test)):
        words, distance = vq(des_list_test[i][1], voc)
        for w in words:
            test_features[i][w] += 1

    test_features = stdslr.transform(test_features)

    return im_features, test_features


"""
    clf = LinearSVC(max_iter=80000)
    clf.fit(im_features, np.array(y_train))

    des_list_test = []

    for image_path in image_paths_test:
        image = cv2.imread(image_path)
        kp = orb.detect(image, None)
        keypoints_test, descriptor_test = orb.compute(image, kp)
        des_list_test.append((image_path, descriptor_test))

    test_features = np.zeros((len(image_paths_test), k), "float32")
    for i in range(len(image_paths_test)):
        words, distance = vq(des_list_test[i][1], voc)
        for w in words:
            test_features[i][w] += 1

    test_features = stdslr.transform(test_features)

    true_classes = []
    for i in y_test:
        if i == 0:
            true_classes.append("Cheetah")
        if i == 1:
            true_classes.append("Jaguar")
        if i == 2:
            true_classes.append("Leopard")
        if i == 3:
            true_classes.append("Lion")
        if i == 4:
            true_classes.append("Tiger")

    predict_classes = []
    for i in clf.predict(test_features):
        if i == 0:
            predict_classes.append("Cheetah")
        if i == 1:
            predict_classes.append("Jaguar")
        if i == 2:
            predict_classes.append("Leopard")
        if i == 3:
            predict_classes.append("Lion")
        if i == 4:
            predict_classes.append("Tiger")

    clf.predict(test_features)
    accuracy = accuracy_score(true_classes, predict_classes)
    print(accuracy)
"""


def augment_images(path):
    if not os.path.exists('BigCatsAugmented'):
        os.mkdir('BigCatsAugmented')

    for cats in os.listdir(path):
        if not os.path.exists('BigCatsAugmented/' + cats):
            os.mkdir('BigCatsAugmented/' + cats)

        imgpath = path + '/' + cats

        for file in os.listdir(imgpath):
            # load the image
            img = load_img(imgpath + '/' + file)
            # convert to numpy array
            data = img_to_array(img)
            # expand dimension to one sample
            samples = expand_dims(data, 0)
            # create image data augmentation generator
            datagen = ImageDataGenerator(horizontal_flip=True, rotation_range=90)
            # prepare iterator
            it = datagen.flow(samples, batch_size=1, save_to_dir='BigCatsAugmented/' + cats, save_prefix='aug',
                              save_format='jpg')
            # generate samples and plot
            for i in range(9):
                # generate batch of images
                batch = it.next()
                # convert to unsigned integers for viewing
                image = batch[0].astype('uint8')
