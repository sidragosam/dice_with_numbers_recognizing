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

img = cv.imread(cv.samples.findFile("dice2.jpg"))
if img is None:
    sys.exit("Could not read the image.")

grayImage = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

new_image = np.zeros(grayImage.shape, grayImage.dtype)

alpha = 1.0 	#	[1.0-3.0]
beta = 100 		#	[0-100]

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


cnts = cv.findContours(blackAndWhiteImage.copy(), cv.RETR_EXTERNAL,
	cv.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
digitCnts = []
for c in cnts:
	# compute the bounding box of the contour
	(x, y, w, h) = cv.boundingRect(c)

	# if the contour is sufficiently large, it must be a digit
	if w >= 15 and (h >= 30 and h <= 40):
		digitCnts.append(c)

displayCnt = None


for c in cnts:
	# approximate the contour
	peri = cv.arcLength(c, True)
	approx = cv.approxPolyDP(c, 0.02 * peri, True)

	# if the contour has four vertices, then we have found
	# the thermostat display
	if len(approx) == 4:
		displayCnt = approx
		break

warped = four_point_transform(grayImage, displayCnt.reshape(4, 2))
output = four_point_transform(edged, displayCnt.reshape(4, 2))

digitCnts = contours.sort_contours(digitCnts,
	method="left-to-right")[0]
digits = []

for c in digitCnts:
	# extract the digit ROI
	(x, y, w, h) = cv.boundingRect(c)
	roi = thresh[y:y + h, x:x + w]

	(roiH, roiW) = roi.shape
	(dW, dH) = (int(roiW * 0.25), int(roiH * 0.15))
	dHC = int(roiH * 0.05)

	segments = [
		((0, 0), (w, dH)),	# top
		((0, 0), (dW, h // 2)),	# top-left
		((w - dW, 0), (w, h // 2)),	# top-right
		((0, (h // 2) - dHC) , (w, (h // 2) + dHC)), # center
		((0, h // 2), (dW, h)),	# bottom-left
		((w - dW, h // 2), (w, h)),	# bottom-right
		((0, h - dH), (w, h))	# bottom
	]
	on = [0] * len(segments)

	for (i, ((xA, yA), (xB, yB))) in enumerate(segments):
		segROI = roi[yA:yB, xA:xB]
		total = cv.countNonZero(segROI)
		area = (xB - xA) * (yB - yA)

		# if the total number of non-zero pixels is greater than
		# 50% of the area, mark the segment as "on"
		if total / float(area) > 0.5:
			on[i] = 1

	# lookup the digit and draw it on the image
	digit = DIGITS_LOOKUP[tuple(on)]
	digits.append(digit)
	cv.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 1)
	cv.putText(output, str(digit), (x - 10, y - 10),
				cv.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

print(u"{}{}.{} \u00b0C".format(*digits))
cv.imshow("Input", edged)
cv.imshow("Output", output)
cv.waitKey(0)