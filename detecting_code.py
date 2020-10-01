#!/usr/bin/env python

import cv2
import numpy as np

class Detector():
    def __init__(self, image):
        self.img = cv2.imread(image, cv2.IMREAD_GRAYSCALE) #query image

        #features
        self.sift = cv2.xfeatures2d.SIFT_create()
        self.kp_image, self.desc_image = self.sift.detectAndCompute(self.img,None)

        # Feature matching
        self.index_params = dict(algorithm=0, trees=5)
        self.search_params = dict()
        self.flann = cv2.FlannBasedMatcher(self.index_params, self.search_params)


    def detect(self,frame):
        grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #train image
        kp_grayframe, desc_grayframe = self.sift.detectAndCompute(grayframe,None)
        matches = self.flann.knnMatch(self.desc_image, desc_grayframe, k=2)
        good_points = []

        for m,n in matches: #m is query image, n in image in train image
            if m.distance < 0.6*n.distance:
                good_points.append(m)

        # keypoint_frame = cv2.drawMatches(self.img, self.kp_image, grayframe,kp_grayframe, good_points, frame)

        # Homography
        if len(good_points) > 8:
            query_pts = np.float32([self.kp_image[m.queryIdx].pt for m in good_points]).reshape(-1,1,2)
            train_pts = np.float32([kp_grayframe[m.trainIdx].pt for m in good_points]).reshape(-1,1,2)

            matrix, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0)
            macthes_mask = mask.ravel().tolist()

            # perspective transform
            h,w = self.img.shape
            pts = np.float32([[0,0], [0,h], [w,h], [w,0]]).reshape(-1,1,2)
            dst = cv2.perspectiveTransform(pts, matrix)

            homography = cv2.polylines(frame, [np.int32(dst)], True, (255,0,0), 3)

            return homography
            # cv2.imshow("Homography", homography)
        else:
            # cv2.imshow("Homography", grayframe)
            return frame



if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    detector = Detector("robot.png")
    while True:
        ret, frame = cap.read()
        frame = detector.detect(frame)
        cv2.imshow("Test", frame)

