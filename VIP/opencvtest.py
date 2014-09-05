import cv2
import numpy as np

image = cv2.imread('Img001_diffuse_smallgray.png')
gray= cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT()
kp = sift.detect(gray,None)

image=cv2.drawKeypoints(gray,kp,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imwrite('sift_keypoints.jpg',image)
