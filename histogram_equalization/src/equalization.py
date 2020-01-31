import numpy as np
import cv2
from matplotlib import pyplot as plt
import math

MAX_VaLUE = 256

def calculate_histogram(img):
    hist = [0] * MAX_VaLUE
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            hist[img.item(i,j)] += 1
    return hist

def calc_cumulative_frequency_distribution(array):
    if len(array):
        cfd = [0] * len(array)
        cfd[0] = array[0]
        for i in range(1, len(array)):
            cfd[i] = cfd[i-1] + array[i]
        return cfd
    return []

def mapping(cfd, img):
    alpha = float(MAX_VaLUE) / (img.shape[0] * img.shape[1])
    mapping = [0] * len(cfd)
    for i in range(len(cfd)):
        mapping[i] = math.floor( cfd[i] * alpha )
    return mapping

# this function returns an image with equalized histogram
def equalization(img):
    eq_img = np.zeros([img.shape[0], img.shape[1], 1], dtype=np.uint8)
    hist = calculate_histogram(img)
    cfd = calc_cumulative_frequency_distribution(hist)
    mp = mapping(cfd, img)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            eq_img[i][j] = mp[img[i,j]]

    return eq_img

FILE_NAME = "/Users/michaelfilonenko/Downloads/8plus_night_01.jpg"

img = cv2.imread(FILE_NAME, 0)
cv2.imshow('0', img)

e_img = equalization(img)
cv2.imshow('1', e_img)

plt.hist(img.ravel(),256,[0,256])
plt.hist(e_img.ravel(),256,[0,256])
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
