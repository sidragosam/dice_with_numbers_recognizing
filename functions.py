import numpy as np
from statistics import mean

def in_range_img(image):
    width, height = image.shape[:2]
    new_img = np.zeros([width, height, 4])

    for i in range(width):
        for j in range(height):
            rgb = [float(image[i][j][0]), float(image[i][j][1]), float(image[i][j][2])]
            avg = float(mean(rgb)/250)
            if avg > 0.2:
                avg = 0
            else:
                avg = 1
            new_img[i][j][0] = avg
            new_img[i][j][1] = avg
            new_img[i][j][2] = avg
            new_img[i][j][3] = 1
    return new_img
