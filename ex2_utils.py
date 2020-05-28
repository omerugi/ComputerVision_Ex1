import math
from math import exp

import cv2.cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage


def conv1D(inSignal: np.ndarray, kernel1: np.ndarray) -> np.ndarray:
    """
    Convolve a 1-D array with a given kernel
    :param inSignal: 1-D array
    :param kernel1: 1-D array as a kernel
    :return: The convolved array
    """
    k = kernel1[::-1]
    a = np.pad(inSignal, (len(k) - 1, len(k) - 1), 'constant')
    res = np.zeros(len(a) - len(k) + 1)
    for i in range(0, len(a) - len(k) + 1):
        res[i] = np.multiply(a[i:i + len(k)], k).sum()

    return res


def conv2D(inImage: np.ndarray, kernel2: np.ndarray) -> np.ndarray:
    """
    Convolve a 2-D array with a given kernel
    :param inImage: 2D image
    :param kernel2: A kernel
    :return: The convolved image
    """


    k = kernel2

    # if the kernel is a vector with shape like ([1,1,1]) pad it with 0's to be a square
    if(len(k.shape)==1):
        k=np.pad(k.reshape(1, len(k)).transpose(), (len(k) // 2, len(k) // 2), 'constant')
        k=k[1:k.shape[0]-1,:]

    #if the kernel is a vector wuth shape like ([[1],[1],[1]) pad it with 0's to be a square
    if(k.shape[1]==1):
        k = np.pad(k.reshape(1, len(k)).transpose(), (len(k) // 2, len(k) // 2), 'constant')
        k = k[1:k.shape[0] - 1, :]

    #pad the image
    a = np.pad(inImage, (k.shape[0] // 2, k.shape[1] // 2), 'edge').astype('float32')

    #the result
    res = np.ndarray(inImage.shape).astype('float32')

    #the convolution
    for i in range(res.shape[0]):
        for j in range(res.shape[1]):
            res[i, j] = (a[i:i + k.shape[0], j:j + k.shape[1]] * k).sum()

    return res


def convDerivative(inImage: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """
    Calculate gradient of an image
    :param inImage: Grayscale iamge
    :return: (directions, magnitude,x_der,y_der)
    """
    #the kernel for the derivative
    k = np.array([[0, 1, 0],
                  [0, 0, 0],
                  [0, -1, 0]])

    Ix = conv2D(inImage, k.transpose())
    Iy = conv2D(inImage, k)
    mag = np.sqrt(np.power(Ix, 2) + np.power(Iy, 2))
    div = np.arctan2(Iy, Ix)
    return div, mag, Ix, Iy


def blurImage1(in_image: np.ndarray, kernel_size: int) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel
    :param inImage: Input image
    :param kernelSize: Kernel size
    :return: The Blurred image
    """

    # Optimal sigma
    sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8

    # Creates the karnel accurding to gausian furmula
    gaussian1 = np.array([1, 1])
    gaussian1D = np.array([1, 1])
    for i in range(0, kernel_size -2):
        gaussian1D = conv1D(gaussian1D, gaussian1)
    gaussian1D = gaussian1D.reshape(kernel_size, 1)
    gaussian = gaussian1D * gaussian1D.transpose()
    gaussian = gaussian / gaussian.sum()

    # Resturns the blurred img
    return conv2D(in_image, gaussian)


def blurImage2(in_image: np.ndarray, kernel_size: int) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel using OpenCV built-in functions
    :param inImage: Input image
    :param kernelSize: Kernel size
    :return: The Blurred image
    """

    sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8
    guassian = cv.getGaussianKernel(kernel_size, sigma)
    guassian = guassian * guassian.transpose()
    return cv.filter2D(in_image, -1, guassian, borderType=cv.BORDER_REPLICATE)


def edgeDetectionSobel(img: np.ndarray, thresh: float = 0.2) -> (np.ndarray, np.ndarray):
    """
    Detects edges using the Sobel method
    :param img: Input image
    :param thresh: The minimum threshold for the edge response
    :return: opencv solution, my implementation
    """
    s = np.array([[1, 0, -1],
                  [2, 0, -2],
                  [1, 0, -1]])

    thresh *= 255
    # my_res = np.sqrt((conv2D(img, s) ** 2 + conv2D(img, s.transpose()) ** 2))
    my_res = np.sqrt((cv.filter2D(img, -1, s, borderType=cv.BORDER_REPLICATE) ** 2 + cv.filter2D(img, -1, s.transpose(),
                                                                                                 borderType=cv.BORDER_REPLICATE) ** 2))
    my = np.ndarray(my_res.shape)
    my[my_res > thresh] = 1
    my[my_res < thresh] = 0

    cv_res = cv.magnitude(cv.Sobel(img, -1, 1, 0), cv.Sobel(img, -1, 0, 1))
    v = np.ndarray(cv_res.shape)
    v[cv_res > thresh] = 1
    v[cv_res < thresh] = 0

    return cv_res, my_res


def edgeDetectionZeroCrossingSimple(img: np.ndarray) -> (np.ndarray):
    """
    Detecting edges using the "ZeroCrossing" method
    :param I: Input image
    :return: Edge matrix
    """
    k = np.array([[0, 1, 0],
                  [1, -4, 1],
                  [0, 1, 0]])
    d = cv.filter2D(img, -1, k, borderType=cv.BORDER_REPLICATE)
    res = np.zeros(d.shape)

    # Check for a zero crossing around (x,y)
    for i in range(0, d.shape[0]):
        for j in range(0, d.shape[1]):
            try:
                if d[i, j] == 0:
                    if (d[i, j + 1] > 0 and d[i, j - 1] < 0) or (d[i, j + 1] < 0 and d[i, j - 1] > 0) or (
                            d[i + 1, j] > 0 and d[i - 1, j] < 0) or (d[i + 1, j] < 0 and d[i - 1, j] > 0):
                        res[i, j] = 1
                elif d[i, j] > 0:
                    if d[i, j + 1] < 0 or d[i + 1, j] < 0:
                        res[i, j] = 1
                else:
                    if d[i, j + 1] > 0 or d[i + 1, j] > 0:
                        res[i, j] = 1

            except IndexError as e:
                pass
    return res


def edgeDetectionZeroCrossingLOG(img: np.ndarray) -> np.ndarray:
    """
    Detecting edges using the "ZeroCrossingLOG" method
    :param I: Input image
    :return: :return: Edge matrix
    """
    blur = blurImage2(img, 5)
    return edgeDetectionZeroCrossingSimple(blur)


def sobleForCanny(img: np.ndarray):
    """
    A simple sobel that uses CV's functions.
    """
    G = np.sqrt(np.power(cv.Sobel(img, -1, 0, 1), 2) + np.power(cv.Sobel(img, -1, 1, 0), 2))
    theta = np.arctan2(cv.Sobel(img, -1, 0, 1), cv.Sobel(img, -1, 1, 0))
    return G, theta


def edgeDetectionCanny(img: np.ndarray, thrs_1: float, thrs_2: float) -> (np.ndarray, np.ndarray):
    """
    Detecting edges usint "Canny Edge" method
    :param img: Input image
    :param thrs_1: T1
    :param thrs_2: T2
    :return: opencv solution, my implementation
    """

    # Sobel edge detection
    mag, div = sobleForCanny(img)
    # NMS on the results from sobel
    nms = non_max_suppression(mag, div)

    # Check all the nms points and mark the ones that are for sure an edge
    # and mark with 150 all the edges that have potential to be an edge.
    for i in range(0, nms.shape[0]):
        for j in range(0, nms.shape[1]):
            try:
                if nms[i][j] <= thrs_2:
                    nms[i][j] = 0
                elif thrs_2 < nms[i][j] < thrs_1:
                    # Check is one of the neighbors is marked as edge
                    neighbor = nms[i - 1:i + 2, j - 1: j + 2]
                    if neighbor.max() < thrs_1:
                        nms[i][j] = 150
                    else:
                        nms[i][j] = 255
                else:
                    nms[i][j] = 255
            except IndexError as e:
                pass

    # Check all the potential points is they are a part of any edge
    for i in range(0, nms.shape[0]):
        for j in range(0, nms.shape[1]):
            try:
                if nms[i][j] == 150:
                    # Check is one of the neighbors is marked as edge
                    neighbor = nms[i - 1:i + 2, j - 1: j + 2]
                    if neighbor.max() < thrs_1:
                        nms[i][j] = 0
                    else:
                        nms[i][j] = 255
            except IndexError as e:
                pass

    cvc = cv.Canny(img.astype(np.uint8), thrs_1, thrs_2)
    return cvc, nms


def non_max_suppression(img: np.ndarray, D: np.ndarray) -> np.ndarray:
    """
    Preforming a non maximum suppuration to a given img using it's direction matrix
    Will first change the radians to degrees and make all between 0-180
    "Quantisize" the image to 4 groups and will check the neighbors according
    The is to make sure we will get the edges with less noise around them
    """
    M, N = img.shape
    Z = np.zeros((M, N), dtype=np.float32)
    # Change to degrees
    angle = np.rad2deg(D)
    # Make all 0-180
    angle[angle < 0] += 180

    # According to the angle checks for each (x,y) if it's intensity is greater or smaller
    for i in range(1, M - 1):
        for j in range(1, N - 1):
            try:
                q = 255
                r = 255
                # angle 0
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    q = img[i, j + 1]
                    r = img[i, j - 1]
                # angle 45
                elif 22.5 <= angle[i, j] < 67.5:
                    q = img[i + 1, j - 1]
                    r = img[i - 1, j + 1]
                # angle 90
                elif 67.5 <= angle[i, j] < 112.5:
                    q = img[i + 1, j]
                    r = img[i - 1, j]
                # angle 135
                elif 112.5 <= angle[i, j] < 157.5:
                    q = img[i - 1, j - 1]
                    r = img[i + 1, j + 1]

                # Check who is greater amount my neighbors
                if (img[i, j] >= q) and (img[i, j] >= r):
                    Z[i, j] = img[i, j]
                else:
                    Z[i, j] = 0

            except IndexError as e:
                pass

    return Z


def houghCircle(img: np.ndarray, min_radius: float, max_radius: float) -> list:
    """
    Find Circles in an image using a Hough Transform algorithm extension
    :param I: Input image
    :param minRadius: Minimum circle radius
    :param maxRadius: Maximum circle radius
    :return: A list containing the detected circles,
    [(x,y,radius),(x,y,radius),...]
    """

    # Do a canny edge detection
    imgc, _ = edgeDetectionCanny(img, 100, 50)

    # Get the deration matrix after sobel
    _, div = sobleForCanny(img)

    tresh = 20
    # 3D array of all to check all the radius from min to max
    hough = np.zeros((imgc.shape[0], imgc.shape[1], max_radius - min_radius))
    list = []

    # for every R is there is a circle in the image
    for r in range(hough.shape[2]):
        for x in range(0, imgc.shape[1]):
            for y in range(0, imgc.shape[0]):
                if imgc[y, x] != 0:
                    try:
                        # Will mark according to the gradient direction the front and rear points as centers of circles
                        a1 = x + (r + min_radius) * np.cos(div[y, x])
                        b1 = y + (r + min_radius) * np.sin(div[y, x])
                        a2 = x - (r + min_radius) * np.cos(div[y, x])
                        b2 = y - (r + min_radius) * np.sin(div[y, x])
                        hough[int(a1), int(b1), r] += 1
                        hough[int(a2), int(b2), r] += 1

                    except IndexError as e:
                        pass

    # Check if the point is over the threshold and should mark as a center of ac circle
    for r in range(hough.shape[2]):
        for x in range(0, img.shape[0]):
            for y in range(0, img.shape[1]):
                if hough[x, y, r] > tresh:
                    list.append((x, y, min_radius + r))

    return list
