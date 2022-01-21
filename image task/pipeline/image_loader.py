""" Code for loading/augmenting images """
import cv2
import os

# example of horizontal shift image augmentation
from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator


def load_images(path):
    img1 = cv2.imread(path + '/Cheetah/animal-africa-wilderness-zoo.jpg')
    img2 = cv2.imread(path + '/Cheetah/cheetah-223734__340.jpg')

    for cats in os.listdir(path):
        if cats == 'ExtractedFeatures':
            continue

        imgpath = path + '/' + cats
        augpath = path + '/ExtractedFeatures/' + cats

        if not os.path.exists(augpath):
            os.mkdir(augpath)

        for file in os.listdir(imgpath):
            sift(imgpath + '/' + file, 150, augpath)


def sift(image, keypoints, augpath):
    img = cv2.imread(image)
    gray_scale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # initialize SIFT object
    sift = cv2.xfeatures2d.SIFT_create(nfeatures=keypoints)

    # detect keypoints
    keypoints, _ = sift.detectAndCompute(img, None)

    # draw keypoints
    sift_image = cv2.drawKeypoints(gray_scale, keypoints, None)

    # cv2.imshow("Features Image", sift_image)
    # cv2.waitKey(0)
    cv2.imwrite(image.replace('BigCats/', 'BigCats/ExtractedFeatures/'), sift_image)


def augment_images(path):
    if not os.path.exists(path + 'Augmented'):
        os.mkdir(path + 'Augmented')

    for cats in os.listdir(path):
        if not os.path.exists(path + 'Augmented/' + cats):
            os.mkdir(path + 'Augmented/' + cats)

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
