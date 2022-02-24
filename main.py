from imutils.perspective import four_point_transform
from imutils import contours
import imutils

import numpy as np

import cv2 as cv
import sys

DIGITS_LOOKUP = {
	(1, 1, 1, 0, 1, 1, 1): 0,
	(0, 0, 1, 0, 0, 1, 0): 1,
	(1, 0, 1, 1, 1, 1, 0): 2,
	(1, 0, 1, 1, 0, 1, 1): 3,
	(0, 1, 1, 1, 0, 1, 0): 4,
	(1, 1, 0, 1, 0, 1, 1): 5,
	(1, 1, 0, 1, 1, 1, 1): 6,
	(1, 0, 1, 0, 0, 1, 0): 7,
	(1, 1, 1, 1, 1, 1, 1): 8,
	(1, 1, 1, 1, 0, 1, 1): 9
}

img = cv.imread(cv.samples.findFile("dice.jpg"))
if img is None:
    sys.exit("Could not read the image.")

grayImage = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

new_image = np.zeros(grayImage.shape, grayImage.dtype)

alpha = 1.0 #	[1.0-3.0]
beta = 100 #		[0-100]

for y in range(grayImage.shape[0]):
    for x in range(grayImage.shape[1]):
            new_image[y,x] = np.clip(alpha*grayImage[y,x] + beta, 0, 255)

(thresh, blackAndWhiteImage) = cv.threshold(new_image, 127, 255, cv.THRESH_BINARY)
#gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#blurred = cv.GaussianBlur(gray, (5, 5), 0)
edged = cv.Canny(new_image, 50, 200, 255)

#cv.imshow("Ablak", grayImage)
cv.imshow("Teszt ablak", edged)
k = cv.waitKey(0)
if k == ord("s"):
    cv.imwrite("dice.jpg", blackAndWhiteImage)
