"""
Color-sensitive Structural Similarity Measure
"""

import cv2
import argparse
import itertools
import functools
import math
import numpy as np


from scipy import signal
from scipy.ndimage.filters import convolve
from skimage.measure import compare_ssim
from colorsys import rgb_to_hsv
from pathlib import Path
from tabulate import tabulate


DEBUG = False


def options():
    parser = argparse.ArgumentParser()
    parser.add_argument('images', nargs='+')
    parser.add_argument('--colors', nargs='+',
        help="colors in HEX format")
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--filter_size', type=int, default=11)
    parser.add_argument('--weight_background', type=float, default=1)
    parser.add_argument('--downsample', type=float, default=None)
    return parser.parse_args()


def _FSpecialGauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function."""
    radius = size // 2
    offset = 0.0
    start, stop = -radius, radius + 1
    if size % 2 == 0:
      offset = 0.5
      stop -= 1
    x, y = np.mgrid[offset + start:stop, offset + start:stop]
    assert len(x) == size
    g = np.exp(-((x**2 + y**2)/(2.0 * sigma**2)))
    return g / g.sum()


def SSIM(img1, img2, max_val=255, filter_size=11,
    filter_sigma=1.5, k1=0.01, k2=0.03, weights=None):
    """Return the Structural Similarity Map between `img1` and `img2`.
  
    This function attempts to match the functionality of ssim_index_new.m by
    Zhou Wang: http://www.cns.nyu.edu/~lcv/ssim/msssim.zip
  
    Arguments:
      img1: Numpy array holding the first RGB image batch.
      img2: Numpy array holding the second RGB image batch.
      max_val: the dynamic range of the images (i.e., the difference between the
        maximum the and minimum allowed values).
      filter_size: Size of blur kernel to use (will be reduced for small images).
      filter_sigma: Standard deviation for Gaussian blur kernel (will be reduced
        for small images).
      k1: Constant used to maintain stability in the SSIM calculation (0.01 in
        the original paper).
      k2: Constant used to maintain stability in the SSIM calculation (0.03 in
        the original paper).
      weights: Numpy array of weights, same size as image, to be used at the
        pooling step.
  
    Returns:
      Pair containing the mean SSIM and contrast sensitivity between `img1` and
      `img2`.
  
    Raises:
      RuntimeError: If input images don't have the same shape or don't have four
        dimensions: [batch_size, height, width, depth].
    """
    if img1.shape != img2.shape:
        raise RuntimeError('Input images must have the same shape (%s vs. %s).',
            img1.shape, img2.shape)

    if DEBUG:
        cv2.imshow("a", img1)
        cv2.imshow("b", img2)
        cv2.waitKey(0)

    height, width = img1.shape

    # Filter size can't be larger than height or width of images.
    size = min(filter_size, height, width)

    # Scale down sigma if a smaller filter size is used.
    sigma = size * filter_sigma / filter_size if filter_size else 0

    # this uses a magic standard dev calculation:
    # see 1) http://matlabtricks.com/post-19/calculating-standard-deviation-using-minimal-memory
    #     2) http://matlabtricks.com/post-20/calculate-standard-deviation-case-of-sliding-window

    if filter_size:
        window = np.reshape(_FSpecialGauss(size, sigma), (size, size))
        mu1 = signal.fftconvolve(img1, window, mode='valid')
        mu2 = signal.fftconvolve(img2, window, mode='valid')
        sigma11 = signal.fftconvolve(img1 * img1, window, mode='valid')
        sigma22 = signal.fftconvolve(img2 * img2, window, mode='valid')
        sigma12 = signal.fftconvolve(img1 * img2, window, mode='valid')
    else:
        # Empty blur kernel so no need to convolve.
        mu1, mu2 = img1, img2
        sigma11 = img1 * img1
        sigma22 = img2 * img2
        sigma12 = img1 * img2

    mu11 = mu1 * mu1
    mu22 = mu2 * mu2
    mu12 = mu1 * mu2
    sigma11 -= mu11
    sigma22 -= mu22
    sigma12 -= mu12

    # Calculate intermediate values used by both ssim and cs_map.
    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2
    v1 = 2.0 * sigma12 + c2
    v2 = sigma11 + sigma22 + c2

    ssim = (((2.0 * mu12 + c1) * v1) / ((mu11 + mu22 + c1) * v2))

    if DEBUG:
        cv2.imshow("ssim", ssim)
        cv2.waitKey(0)


    if weights is not None:
        
        padding = math.floor(size/2)
        weights = weights[padding:-padding, padding:-padding]

        if DEBUG:
            cv2.imshow("weighted ssim", ssim*weights)
            cv2.waitKey(0)

        ssim = np.average(ssim, weights=weights)
    else:
        ssim = np.mean(ssim)

    cs = np.mean(v1 / v2)
    return ssim, cs


def hex2rgb(hex_):
    hex_ = hex_.lstrip('#')
    return tuple(int(hex_[i:i+2], 16) for i in (0, 2 ,4))


def hex2hsv(hex_):
    rgb = hex2rgb(hex_)
    rgb = [c/255 for c in rgb] # scale to [0,1]
    hsv = rgb_to_hsv(*rgb)

    # For HSV, Hue range is [0,179], Saturation range is [0,255] and Value range is [0,255]
    hsv = (hsv[0]*179, hsv[1]*255, hsv[2]*255) # scale _from_ [0,1]
    return hsv


def extract_hue(color, img):
    """
    Arguments:
      img: an image in the HSV color space
      color: a color in the HSV space
    """
    h_tresh = 20
    
    # For HSV, Hue range is [0,179], Saturation range is [0,255] and Value range is [0,255]
    min_hsv = np.array([max(0.0, color[0] - h_tresh),  10.0,   0.0])
    max_hsv = np.array([min(179.0, color[0] + h_tresh), 250.0, 255.0])

    mask = cv2.inRange(img, min_hsv, max_hsv)
    result = cv2.bitwise_and(img, img, mask = mask)

    return cv2.cvtColor(result, cv2.COLOR_HSV2BGR)

def extract_hue_range(min_hue, max_hue, img):
    """
    Arguments:
      min_hue: min hue (HSV), from 0 to 179
      max_hue: max hue (HSV), from 0 to 179
    """

    # For HSV, Hue range is [0,179], Saturation range is [0,255] and Value range is [0,255]
    min_hsv = np.array([min_hue,  0.0,   0.0])
    max_hsv = np.array([max_hue, 255.0, 255.0])

    mask = cv2.inRange(img, min_hsv, max_hsv)
    result = cv2.bitwise_and(img, img, mask = mask)

    # cv2.imshow("layer", cv2.cvtColor(result, cv2.COLOR_HSV2BGR))
    # cv2.waitKey()

    return cv2.cvtColor(result, cv2.COLOR_HSV2BGR)


def scale(img, width, height):
    return cv2.resize(img, (width, height), interpolation = cv2.INTER_NEAREST)


if __name__ == '__main__':
    opts = options()

    DEBUG = opts.debug

    background = "white"
    compare_background = True

    # split the Hue space into 4 layers, the first holds the background (white or black)
    n_layers = 4 # number of layers in addition to background
    hue_range_size = round(179/n_layers) 
    hue_ranges = [(i*hue_range_size, j*hue_range_size - 1) \
        for i, j in zip(range(n_layers), range(1,n_layers+1))]

    if background == "white":
        hue_ranges = [(i+1, j+1) for i, j in hue_ranges]
        if compare_background: hue_ranges.insert(0, (0,0))
    elif background == "black":
        hue_ranges = [(i-1, j-1) for i, j in hue_ranges]
        if compare_background: hue_ranges.insert(0, (179,179))

    if compare_background: n_layers += 1

    print(hue_ranges)

    filter_size = opts.filter_size
    weight_background = opts.weight_background
    image_paths = opts.images
    image_names = [Path(p).stem for p in image_paths]
    images = [cv2.imread(path, cv2.IMREAD_UNCHANGED) for path in image_paths]
    images = [cv2.cvtColor(img, cv2.COLOR_BGR2HSV) for img in images]

    # Scale all images to lowest size
    width  = min(map(lambda x: x.shape[1], images))
    height = min(map(lambda x: x.shape[0], images))

    if opts.downsample:
        width  = round(opts.downsample * width)
        height = round(opts.downsample * height)

    images = [scale(img, width, height) for img in images]

    # colors_hex = opts.colors
    # colors = list(map(hex2hsv, colors_hex))

    color_layers = [] # a list of layers for each image
    foreground_masks = [] # foreground pixels == TRUE

    for img in images:        
        layers = [cv2.cvtColor(extract_hue_range(min_hue, max_hue, img), cv2.COLOR_BGR2GRAY) \
            for min_hue, max_hue in hue_ranges]
        # extract location of 0-value pixels (background)
        masks = [layer != 0 for layer in layers] 

        np.set_printoptions(threshold=np.nan)
        # print(layers[0])

        color_layers.append(layers)
        foreground_masks.append(masks)

    scores = []

    for i, j in itertools.combinations(range(len(images)), 2):
        print("-----------")
        print("{} x {}".format(image_names[i], image_names[j]))
        A_layers = color_layers[i]
        B_layers = color_layers[j]

        ssim_layers = []

        for k in range(n_layers):
            weights = np.logical_or(foreground_masks[i][k], foreground_masks[j][k]).astype(np.float)
            if np.sum(weights) == 0:
                continue

            # cv2.imshow("mask_A", foreground_masks[i][k].astype(np.float))
            # cv2.imshow("mask_B", foreground_masks[j][k].astype(np.float))
            # cv2.waitKey(0)


            window = np.reshape(_FSpecialGauss(filter_size, 1.5), (filter_size, filter_size))
            weights = signal.fftconvolve(weights, window, mode='same')
            weights[weights==0] = weight_background

            if DEBUG:
                cv2.imshow("weights", weights)
                cv2.waitKey(0)

            ssim, cs = SSIM(A_layers[k], 
                B_layers[k], 
                weights=weights
                # filter_size=filter_size
                )

            ssim_layers.append(ssim)
            
            print("SSIM - {}: {}".format(hue_ranges[k], ssim))

        score = np.mean(ssim_layers)
        print("Mean SSIM: {}".format(score))

        scores.append(
            (Path(image_paths[i]).stem,
            Path(image_paths[j]).stem,
            score)
        )

    print()
    print(tabulate(scores, headers=["imageA", "imageB", "CS-SSIM"]))
