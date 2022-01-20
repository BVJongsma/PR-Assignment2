""" Code for loading/augmenting images """
import cv2
import os


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

    #cv2.imshow("Features Image", sift_image)
    #cv2.waitKey(0)
    cv2.imwrite(image.replace('BigCats/', 'BigCats/ExtractedFeatures/'), sift_image)
