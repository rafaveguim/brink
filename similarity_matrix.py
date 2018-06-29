#!/usr/bin/env python3

import sys
import cv2
import numpy as np
import skimage.measure as measures

import argparse
import pandas as pd
import plotnine

from pathlib import Path

import itertools
from math import floor, ceil
from tqdm import tqdm

def options():
    parser = argparse.ArgumentParser()
    parser.add_argument('images', nargs='+')
    parser.add_argument('--csv', action='store_true')
    parser.add_argument('--rotation_invariant', action='store_true')
    return parser.parse_args()


def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err


def compare(imageA, imageB, measure=None):
    if not measure:
        measure = measures.compare_ssim
    return measure(imageA, imageB)

def pad_images(imgA, imgB):
    # add padding so that images have the same size (if needed)
    width  = lambda x: x.shape[1]
    height = lambda x: x.shape[0]

    # print(imgA.shape)
    # print(imgB.shape)

    # split padding between right/left or top/bottom
    split = lambda v: (v/2, v/2) if v % 2 == 0 else (floor(v/2), ceil(v/2))

    if width(imgA) != width(imgB):
        argminW = min(imgA, imgB, key=width)
        maxW = max(width(imgA), width(imgB))

        diff = maxW - width(argminW)
        padding_left, padding_right = split(diff)
        img_padded = cv2.copyMakeBorder(argminW, 0, 0,
            int(padding_left),
            int(padding_right),
            cv2.BORDER_CONSTANT,
            None,
            255)

        if maxW == width(imgA):
            imgB = img_padded
        else:
            imgA = img_padded

    if height(imgA) != height(imgB):
        argminH = min(imgA, imgB, key=height)
        maxH = max(height(imgA), height(imgB))

        diff = maxH - height(argminH)
        padding_bottom, padding_top = split(diff)
        img_padded = cv2.copyMakeBorder(argminH,
            int(padding_bottom),
            int(padding_top),
            0, 0,
            cv2.BORDER_CONSTANT,
            None,
            255)

        if maxH == height(imgA):
            imgB = img_padded
        else:
            imgA = img_padded

    # cv2.imshow('ImageA', imgA)
    # cv2.imshow('ImageB', imgB)
    # input("Proceed")
    return (imgA, imgB)

def pad_image(img, w, h):
        # add padding so that images have the same size (if needed)
    width  = lambda x: x.shape[1]
    height = lambda x: x.shape[0]

    # split padding between right/left or top/bottom
    split = lambda v: (v/2, v/2) if v % 2 == 0 else (floor(v/2), ceil(v/2))

    if width(img) < w:
        diff = w - width(img)
        padding_left, padding_right = split(diff)
        img = cv2.copyMakeBorder(img, 0, 0,
            int(padding_left),
            int(padding_right),
            cv2.BORDER_CONSTANT,
            None,
            255)

    if height(img) < h:
        diff = h - height(img)
        padding_top, padding_bottom = split(diff)
        img = cv2.copyMakeBorder(img,
            int(padding_top),
            int(padding_bottom),
            0, 0,
            cv2.BORDER_CONSTANT,
            None,
            255)

    # print(width(img), height(img))
    return img

def rotate(img, angle):
    cols, rows = img.shape
    M = cv2.getRotationMatrix2D((cols/2,rows/2), angle, 1)
    return cv2.warpAffine(img, M, (cols, rows))


def scale(img, width, height):
    return cv2.resize(img, (width, height), interpolation = cv2.INTER_CUBIC)

if __name__ == '__main__':
    opts = options()

    # Load images
    images = []

    for image_path in opts.images:
        image_path = Path(image_path)

        img = cv2.imread(str(image_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        images.append((image_path, img))


    # # Pad all images relative to the largest one
    # max_width  = max(map(lambda x: x[1].shape[1], images))
    # max_height = max(map(lambda x: x[1].shape[0], images))

    # Scale all images to lowest size
    width  = min(map(lambda x: x[1].shape[1], images))
    height = min(map(lambda x: x[1].shape[0], images))

    # if rotation invariant, we want all images to be a perfect square
    if opts.rotation_invariant:
        width = height = max(width, height)

    images = [(imgpath, scale(img, width, height)) for imgpath, img in tqdm(images)]

    if opts.rotation_invariant:
        rotations = dict()
        for path, img in images:
            rotations[path] = (rotate(img,  90),
                               rotate(img, 180),
                               rotate(img, 270))

    records = []

    for a, b in tqdm(list(itertools.combinations(images, 2))):
        imgA_path, imgA = a
        imgB_path, imgB = b

        ssim = compare(imgA, imgB, measures.compare_ssim)

        if opts.rotation_invariant:
            imgB_90, imgB_180, imgB_270 = rotations[imgB_path]

            # cv2.imshow('refImage', imgA)
            # cv2.imshow('90', imgB_90)
            # cv2.imshow('180', imgB_180)
            # cv2.imshow('270', imgB_270)

            ssim_90  = compare(imgA, imgB_90, measures.compare_ssim)
            ssim_180 = compare(imgA, imgB_180, measures.compare_ssim)
            ssim_270 = compare(imgA, imgB_270, measures.compare_ssim)

            ssim = max(ssim, ssim_90, ssim_180, ssim_270)

            # print('best score ', ssim)
            # input("Proceed")

        records.append((imgA_path.name,
            imgB_path.name,
            ssim))

    df = pd.DataFrame.from_records(records, columns=['A', 'B', 'ssim_distance'])
    if opts.csv:
        df.to_csv(sys.stdout, index=False)
    else:
        print(df.to_string())


