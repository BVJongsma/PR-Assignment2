import cv2
import os
import numpy as np

from scipy.cluster.vq import kmeans, vq
from sklearn.preprocessing import StandardScaler
from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator


def img_list(path):
    return (os.path.join(path, f) for f in os.listdir(path))


# load images found in the path given in train_path and apply feature extraction
def load_images(train_path, method, features=500):
    class_names = os.listdir(train_path)
    image_paths = []
    image_classes = []
    i = 0

    # load the images and assign labels based on the folder that these images are in
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

    image_paths, labels = zip(*dataset)

    features = extraction(image_paths, method, features)

    return features, list(labels)


# feature extraction using either ORB or SIFT
def extraction(image_paths, method, features):
    des_list = []

    if method == 'orb':
        feature_extractor = cv2.ORB_create(nfeatures=features)
    elif method == 'sift':
        feature_extractor = cv2.xfeatures2d.SIFT_create(nfeatures=features)

    for image_path in image_paths:
        im = cv2.imread(image_path)
        kp = feature_extractor.detect(im, None)
        keypoints, descriptor = feature_extractor.compute(im, kp)
        des_list.append((image_path, descriptor))
    descriptors = des_list[0][1]

    for image_path, descriptor in des_list[1:]:
        descriptors = np.vstack((descriptors, descriptor))

    descriptors_float = descriptors.astype(float)

    k = 50
    voc, variance = kmeans(descriptors_float, k, 1)
    im_features = np.zeros((len(image_paths), k), "float32")

    for i in range(len(image_paths)):
        words, distance = vq(des_list[i][1], voc)
        for w in words:
            im_features[i][w] += 1

    stdslr = StandardScaler().fit(im_features)
    im_features = stdslr.transform(im_features)

    return im_features

#producing augmented images
def augment_images(path):
    if not os.path.exists('BigCatsAugmented'):
        os.mkdir('BigCatsAugmented')

    for cats in os.listdir(path):
        if not os.path.exists('BigCatsAugmented/' + cats):
            os.mkdir('BigCatsAugmented/' + cats)

        # don't augment images if files are already present
        if len(os.listdir('BigCatsAugmented/' + cats)) > 0:
            continue

        imgpath = path + '/' + cats

        for file in os.listdir(imgpath):
            # load the image
            img = load_img(imgpath + '/' + file)
            # convert to numpy array
            data = img_to_array(img)
            # expand dimension to one sample
            samples = expand_dims(data, 0)
            # create image data augmentation generator
            datagen = ImageDataGenerator(rotation_range=90)
            # prepare iterator
            it = datagen.flow(samples, batch_size=1, save_to_dir='BigCatsAugmented/' + cats, save_prefix='aug',
                              save_format='jpg')
            # generate samples and plot
            for i in range(9):
                # generate batch of images
                batch = it.next()
                # convert to unsigned integers for viewing
                image = batch[0].astype('uint8')
