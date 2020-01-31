import numpy as np
import cv2
import math;

def apply_filter(img, kernel):
    dim = 3
    filtered_img = np.zeros([img.shape[0], img.shape[1], 1], dtype=np.uint8)
    for i in range(math.floor(dim / 2), math.floor(img.shape[0] - dim / 2)):
        for j in range(math.floor(dim / 2), math.floor(img.shape[1] - dim / 2)):
            result = 0
            for x in range(0,dim):
                for y in range(0,dim):
                    dx = math.floor(i + x - dim / 2)
                    dy = math.floor(j + y - dim / 2)
                    result += img.item(dx, dy, 0) * kernel[x][y]
            filtered_img.itemset((i,j,0), result)
    return filtered_img

def join(first, second):
    joined_img = np.zeros([img.shape[0], img.shape[1], 1], dtype=np.uint8)
    for i in range(first.shape[0]):
        for j in range(img.shape[1]):
            joined_img[i][j] = math.sqrt(first[i][j] * first[i][j] + second[i][j] * second[i][j])
    return joined_img

IM_PATH = "/Users/michaelfilonenko/python/sobel_operator/Bikesgray.jpg"

img = cv2.imread(IM_PATH)
cv2.imshow('original', img)

gx = [[-1.0, 0.0, 1.0],
    [-2.0, 0.0, 2.0],
    [-1.0, 0.0, 1.0]]

gy = [[-1.0, -2.0, -1.0],
    [0.0, 0.0, 0.0],
    [1.0, 2.0, 1.0]]

filter_x = apply_filter(img, gx)
cv2.imshow('1', filter_x)

filter_y = apply_filter(img, gy)
cv2.imshow('2', filter_y)

filtered_img = join(filter_x,filter_y)
cv2.imshow('3', filtered_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
