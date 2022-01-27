""" Code in this file is used to create plots/visuals for the report """
import cv2
import os

if __name__ == '__main__':
    # load image
    img = 'BigCats/Cheetah/animal-africa-wilderness-zoo.jpg'
    image = cv2.imread(img)

    # convert to grayscale image
    gray_scale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if not os.path.exists('plotsVisuals'):
        os.mkdir('plotsVisuals')
    if not os.path.exists('plotsVisuals/sift'):
        os.mkdir('plotsVisuals/sift')
    if not os.path.exists('plotsVisuals/orb'):
        os.mkdir('plotsVisuals/orb')

    for i in [50,100,150,200,250,500]:
        # initialize SIFT object
        sift = cv2.xfeatures2d.SIFT_create(nfeatures=i)

        # detect keypoints
        keypoints, _ = sift.detectAndCompute(image, None)

        # draw keypoints
        sift_image = cv2.drawKeypoints(gray_scale, keypoints, None)

        cv2.imwrite("plotsVisuals/sift/" + str(i) + "keypoints.jpg", sift_image)

    for i in [50, 100, 150, 200, 250, 500]:
        # initialize SIFT object
        orb = cv2.ORB_create(nfeatures=i)

        # detect keypoints
        keypoints, _ = sift.detectAndCompute(image, None)

        # draw keypoints
        orb_image = cv2.drawKeypoints(gray_scale, keypoints, None)

        cv2.imwrite("plotsVisuals/orb/" + str(i) + "keypoints.jpg", orb_image)
