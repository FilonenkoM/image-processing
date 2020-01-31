import cv2;
import math
import numpy as np

from canny_operator import canny_filter

IM_PATH =  "/Users/michaelfilonenko/LocalDocs/Screenshots/Снимок экрана 2019-03-27 в 19.46.22.png"

# def apply_filter(img: np.ndarray, filter: np.ndarray, n=1) -> np.ndarray:
#     filtered_img = np.zeros([img.shape[0], img.shape[1], 1], dtype=np.int64)
#     for i in range(1, img.shape[0] - 1):
#         for j in range(1, img.shape[1] - 1):
#             sum = 0;
#             for x in range(0, 3):
#                 for y in range(0, 3):
#                      sum += img.item(i + x - 1, j + y - 1) * filter.item(x, y)
#             filtered_img.itemset((i, j, 0), sum)
#     return filtered_img

def gaussian_filter(img):
    dim = 5
    filtered_img = img.copy()
    kernel = [[2, 4, 5, 4, 2],
              [4, 9, 12, 9, 4],
              [5, 12, 15, 12, 5],
              [4, 9, 12, 9, 4],
              [2, 4, 5, 4, 2]]
    for i in range(math.floor(dim / 2), math.floor(img.shape[0] - dim / 2)):
        for j in range(math.floor(dim / 2), math.floor(img.shape[1] - dim / 2)):
            result = 0
            for x in range(0,dim):
                for y in range(0,dim):
                    dx = math.floor(i + x - dim / 2)
                    dy = math.floor(j + y - dim / 2)
                    result += img[dx,dy] * kernel[x][y]
            result /= np.sum(kernel)
            filtered_img.itemset((i,j), result)
    return filtered_img

def apply_filter(img: np.ndarray, filter: np.ndarray, n=1) -> np.ndarray:
    filtered_img = np.zeros([img.shape[0], img.shape[1], 1], dtype=np.int64)
    for i in range(1, img.shape[0] - 1):
        for j in range(1, img.shape[1] - 1):
            sum = 0;
            for x in range(0, 3):
                for y in range(0, 3):
                     sum += img[i + x - 1, j + y - 1] * filter[x, y]
            filtered_img[i,j] = sum
    return filtered_img

def join(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    filtered_img = np.zeros([first.shape[0], first.shape[1], 1], dtype=np.uint8)
    for i in range(0, first.shape[0]):
        for j in range(0, first.shape[1]):
            pixel = math.sqrt(first.item(i,j,0) * first.item(i, j, 0) + second.item(i, j, 0) * second.item(i, j, 0))
            filtered_img.itemset((i,j,0), pixel // 4)
    return filtered_img

def sobel_derivatives(img):
    fx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]])

    fy = np.array([[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]])

    fx_img = apply_filter(img, fx)
    fy_img = apply_filter(img, fy)
    return (fx_img, fy_img)

def to_binary(img):
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if(img[i,j] > 100):
                img[i,j] = 255
            else:
                img[i,j] = 0

def harris_corner_detector(img):
    # b_img = canny_filter

    b_img = gaussian_filter(img)

    cv2.imshow("canny", b_img)

    der = sobel_derivatives(b_img)

    Ixx = der[0] ** 2
    Ixy = der[1] * der[0]
    Iyy = der[1] ** 2

    window_size = 6
    offset = window_size // 2
    k = 0.18
    harris_response = []

    for y in range(offset, img.shape[0] - offset):
        for x in range(offset, img.shape[1] - offset):
            Sxx = np.sum(Ixx[y - offset:y + 1 + offset, x - offset:x + 1 + offset])
            Syy = np.sum(Iyy[y - offset:y + 1 + offset, x - offset:x + 1 + offset])
            Sxy = np.sum(Ixy[y - offset:y + 1 + offset, x - offset:x + 1 + offset])

            # Find determinant and trace, use to get corner response
            det = (Sxx * Syy) - (Sxy ** 2)
            trace = Sxx + Syy
            r = det - k * (trace ** 2)

            harris_response.append([x, y, r])

    copy = img.copy()
    for response in harris_response:
        x, y, r = response
        if r > 1000000000:
            print(r)
            copy[y, x] = 255

    cv2.imshow("copy", copy)
    cv2.waitKey(0)

    return copy

if __name__ == "__main__":
    # img = cv2.imread(IM_PATH, 0)
    # cv2.imshow("original", img)
    # # cv2.imshow("original", img)
    # # cv2.imshow("harris corner detector", rgimg)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    im = cv2.imread(IM_PATH, 0)
    cv2.imshow("original", im)
    rgimg = harris_corner_detector(im)
    cv2.imshow("Filtered", rgimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
