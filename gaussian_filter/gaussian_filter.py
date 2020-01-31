
import numpy as np
import cv2
import math

FILE_NAME = "/Users/michaelfilonenko/Downloads/Geneva.tif"

def gaussian(x,y,dim,sigma):
    result =  math.exp( - (x * x + y * y) / (2 * sigma * sigma)) / (2 * math.pi * sigma * sigma)
    return result

def calc_kernel(dim, sigma):
    kernel = [[0.0 for x in range(dim)] for y in range(dim)]
    d = math.floor(dim / 2)
    for i in range(dim):
        for j in range(dim):
            x = i - 2
            y = j - 2
            kernel[i][j] = gaussian(x,y,dim,sigma)
    return kernel

def gaussian_filter(img):
    dim = 3
    filtered_img = np.zeros([img.shape[0], img.shape[1], 1], dtype=np.uint8)
    kernel = calc_kernel(dim, 1.0)
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
                    result += img.item(dx, dy) * kernel[x][y]
            result /= np.sum(kernel)
            filtered_img.itemset((i,j,0), result)
    return filtered_img

def filter(img):
    dim = 3
    filtered_img = np.zeros([img.shape[0], img.shape[1], 1], dtype=np.uint8)
    kernel = calc_kernel(dim, 1.0)
    for i in range(math.floor(dim / 2), math.floor(img.shape[0] - dim / 2)):
        for j in range(math.floor(dim / 2), math.floor(img.shape[1] - dim / 2)):
            result = 0
            for x in range(0,dim):
                for y in range(0,dim):
                    dx = math.floor(i + x - dim / 2)
                    dy = math.floor(j + y - dim / 2)
                    result += img.item(dx, dy) * kernel[x][y]
            result /= np.sum(kernel)
            filtered_img.itemset((i,j,0), result)
    return filtered_img

img = cv2.imread(FILE_NAME, 0)
cv2.imshow('0', img)

gaussian_img = gaussian_filter(img)
cv2.imshow("1", gaussian_img)

filtered_img = filter(img)
cv2.imshow("2", filtered_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
