import cv2;
import numpy as np;
import math;

import unittest

IM_PATH =  "/Users/michaelfilonenko/Downloads/8plus_night_01.jpg"

def gaussian_filter(img):
    dim = 3
    filtered_img = np.zeros([img.shape[0], img.shape[1], 1], dtype=np.uint8)
    kernel = [[1,2,1],
        [2,4,2],
        [1,2,1]]
    for i in range(math.floor(dim / 2), math.floor(img.shape[0] - dim / 2)):
        for j in range(math.floor(dim / 2), math.floor(img.shape[1] - dim / 2)):
            result = 0
            for x in range(0,dim):
                for y in range(0,dim):
                    dx = math.floor(i + x - dim / 2)
                    dy = math.floor(j + y - dim / 2)
                    result += img.item(dx, dy, 0) * kernel[x][y]
            result /= np.sum(kernel)
            filtered_img.itemset((i,j,0), result)
    return filtered_img

def apply_filter(img: np.ndarray, filter: np.ndarray, n=1) -> np.ndarray:
    filtered_img = np.zeros([img.shape[0], img.shape[1], 1], dtype=np.int64)
    for i in range(1, img.shape[0] - 1):
        for j in range(1, img.shape[1] - 1):
            sum = 0;
            for x in range(0, 3):
                for y in range(0, 3):
                     sum += img.item(i + x - 1, j + y - 1, 0) * filter.item(x, y)
            filtered_img.itemset((i, j, 0), sum)
    return filtered_img

def join(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    filtered_img = np.zeros([first.shape[0], first.shape[1], 1], dtype=np.uint8)
    for i in range(0, first.shape[0]):
        for j in range(0, first.shape[1]):
            pixel = math.sqrt(first.item(i,j,0) * first.item(i, j, 0) + second.item(i, j, 0) * second.item(i, j, 0))
            # if pixel < 100:
            #     pixel = 0
            filtered_img.itemset((i,j,0), pixel)
    return filtered_img

def join_abs(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    filtered_img = np.zeros([first.shape[0], first.shape[1], 1], dtype=np.uint8)
    for i in range(0, first.shape[0]):
        for j in range(0, first.shape[1]):
            pixel = abs(first.item(i, j, 0)) + abs(second.item(i, j, 0))
            # if pixel < 100:
            #     pixel = 0
            filtered_img.itemset((i,j,0), pixel)
    return filtered_img

img = cv2.imread(IM_PATH)
cv2.imshow('original', img)

img = gaussian_filter(img)
img = gaussian_filter(img)

fx = np.array([[-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]])

fy = np.array([[-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]])

fx_img = apply_filter(img, fx)

fy_img = apply_filter(img, fy)

filtered_img = join(fx_img, fy_img)
cv2.imshow('result', filtered_img)

filtered_img = join_abs(fx_img, fy_img)
cv2.imshow('result abs', filtered_img)

cv2.waitKey(0)
cv2.destroyAllWindows()

def apply_filter_works():
    test_mat = np.zeros((3,3,1), np.uint8)
    test_mat[0][0] = 0;
    test_mat[0][1] = 0;
    test_mat[0][2] = 255;
    test_mat[1][0] = 0;
    test_mat[1][1] = 0;
    test_mat[1][2] = 255;
    test_mat[2][0] = 0;
    test_mat[2][1] = 0;
    test_mat[2][2] = 255;
    apply_filter(test_mat, fx)





