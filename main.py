# from imutils.perspective import four_point_transform
# from imutils import contours
# import imutils
# from imutils import paths

# from python_imagesearch import PyImageSearchANPR
import numpy as np
# import argparse

from PIL import Image

import cv2 as cv
import sys
import pytesseract

import functions

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

pytesseract.pytesseract.tesseract_cmd = r'C:\Users\hallgato\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

img = cv.imread(cv.samples.findFile("dice2.jpg"))
if img is None:
    sys.exit("Could not read the image.")

grayImage = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
(thresh2, blackAndWhiteImage2) = cv.threshold(grayImage, 127, 255, cv.THRESH_BINARY)
edged2 = cv.Canny(grayImage, 50, 200, 255)
cv.imshow("Edged2", edged2)
new_image = np.zeros(grayImage.shape, grayImage.dtype)

alpha = 1.0  # [1.0-3.0]
beta = 100  # [0-100]

for y in range(grayImage.shape[0]):
    for x in range(grayImage.shape[1]):
        new_image[y, x] = np.clip(alpha * grayImage[y, x] + beta, 0, 255)

(thresh, blackAndWhiteImage) = cv.threshold(new_image, 127, 255, cv.THRESH_BINARY)
# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# blurred = cv.GaussianBlur(gray, (5, 5), 0)
edged = cv.Canny(new_image, 50, 200, 255)

# cv.imshow("Ablak", grayImage)
cv.imshow("Edged", edged)

# thresh = cv.threshold(new_image, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (1, 5))
thresh = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel)


def inverte(imagem):
    imagem = cv.bitwise_not(imagem)
    return imagem


cv.imshow("Threshold", thresh)

newthresh = functions.in_range_img(img)

cv.imshow("Uj threshold", newthresh)

# Read image
im_in = cv.imread("dice6.jpg", cv.IMREAD_GRAYSCALE)

# Threshold.
# Set values equal to or above 220 to 0.
# Set values below 220 to 255.

th, im_th = cv.threshold(im_in, 220, 255, cv.THRESH_BINARY_INV)

# Copy the thresholded image.
im_floodfill = im_th.copy()

# Mask used to flood filling.
# Notice the size needs to be 2 pixels than the image.
h, w = im_th.shape[:2]
mask = np.zeros((h+2, w+2), np.uint8)

# Floodfill from point (0, 0)
cv.floodFill(im_floodfill, mask, (0, 0), 255)

# Invert floodfilled image
im_floodfill_inv = cv.bitwise_not(im_th)

# Combine the two images to get the foreground.
im_out = im_th  # | im_floodfill_inv

# Display images.
cv.imshow("Thresholded Image", im_th)
cv.imshow("Floodfilled Image", im_floodfill)
cv.imshow("Inverted Floodfilled Image", im_floodfill_inv)
cv.imshow("Foreground", im_out)
# cv.waitKey(0)

finalimg = cv.imread("dice2.jpg")
finalthresh = functions.in_range_img(finalimg)

cv.imshow("FinalIMG", finalthresh)


options = "outputbase digits"
custom_config = r'--oem 3 --psm 6 outputbase digits -c tessedit_char_whitelist=123456'

im = Image.fromarray((finalthresh * 255).astype(np.uint8))  # átalakítás 0-1-ről 0-255-re
print("Számok: " + pytesseract.image_to_string(im, config=custom_config))
cv.imshow("Output", im_out)

# rgb = cv.cvtColor(invertalt, cv.COLOR_BGR2RGB)
# text = pytesseract.image_to_string(im_floodfill_inv, config=options)# im_out, config=options)
# print(text)
# cv.imshow("Inverted",invertalt)
cv.waitKey(0)
