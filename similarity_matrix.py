#!/usr/bin/env python3

import sys
import cv2
import numpy as np
import skimage.measure as measures

import argparse
import pandas as pd

from pathlib import Path
from multiprocessing.pool import Pool

import itertools
from math import floor, ceil, factorial
from tqdm import tqdm

def options():
    parser = argparse.ArgumentParser()
    parser.add_argument('images', nargs='+')
    parser.add_argument('--csv', action='store_true')
    parser.add_argument('--rotation_invariant', action='store_true')
    parser.add_argument('--parallel', action='store_true')
    parser.add_argument('--num_processes', type=int, default=4)
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

def do_singleprocess(images, is_rotation_invariant=False):
    if is_rotation_invariant:
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

        if is_rotation_invariant:
            imgB_90, imgB_180, imgB_270 = rotations[imgB_path]

            ssim_90  = compare(imgA, imgB_90, measures.compare_ssim)
            ssim_180 = compare(imgA, imgB_180, measures.compare_ssim)
            ssim_270 = compare(imgA, imgB_270, measures.compare_ssim)

            ssim = max(ssim, ssim_90, ssim_180, ssim_270)

        records.append((imgA_path.name,
            imgB_path.name,
            ssim))

    return records

class Doer():
    def __init__(self, is_rotation_invariant=False, rotations=None):
        self.is_rotation_invariant = is_rotation_invariant
        self.rotations = rotations

    def do(self, in_tuple):
        (imgA_path, imgA), (imgB_path, imgB) = in_tuple

        ssim = compare(imgA, imgB, measures.compare_ssim)

        if self.is_rotation_invariant:
            imgB_90, imgB_180, imgB_270 = self.rotations[imgB_path]

            ssim_90  = compare(imgA, imgB_90, measures.compare_ssim)
            ssim_180 = compare(imgA, imgB_180, measures.compare_ssim)
            ssim_270 = compare(imgA, imgB_270, measures.compare_ssim)

            ssim = max(ssim, ssim_90, ssim_180, ssim_270)

        return imgA_path.name, imgB_path.name, ssim


# def do_multiprocess(images, num_processes, is_rotation_invariant=False):
#     if is_rotation_invariant:
#         rotations = dict()
#         for path, img in images:
#             rotations[path] = (rotate(img,  90),
#                                rotate(img, 180),
#                                rotate(img, 270))

#     pool = Pool(num_processes)
#     doer = Doer(is_rotation_invariant, rotations)

#     records = pool.map(doer.do, list(itertools.combinations(images, 2)))

#     return records

def do_multiprocess(images, num_processes, is_rotation_invariant=False):
    if is_rotation_invariant:
        rotations = dict()
        for path, img in images:
            rotations[path] = (rotate(img,  90),
                               rotate(img, 180),
                               rotate(img, 270))

    pool = Pool(num_processes)
    doer = Doer(is_rotation_invariant, rotations)

    n_images = len(images)
    n_combinations = factorial(n_images) / (factorial(2) * factorial(n_images-2) )

    records=[]
    with tqdm(total=n_combinations) as pbar:
        for record in tqdm(pool.imap_unordered(doer.do, list(itertools.combinations(images, 2)), chunksize=50)):
            records.append(record)
            pbar.update()

    return records


if __name__ == '__main__':
    opts = options()

    # Load images
    images = []

    for image_path in opts.images:
        image_path = Path(image_path)

        img = cv2.imread(str(image_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        images.append((image_path, img))

    # Scale all images to lowest size
    width  = min(map(lambda x: x[1].shape[1], images))
    height = min(map(lambda x: x[1].shape[0], images))

    # if rotation invariant, we want all images to be a perfect square
    if opts.rotation_invariant:
        width = height = max(width, height)

    images = [(imgpath, scale(img, width, height)) for imgpath, img in tqdm(images)]

    if opts.parallel:
        records = do_multiprocess(images, opts.num_processes, opts.rotation_invariant)
    else:
        records = do_singleprocess(images, opts.rotation_invariant)
        

    df = pd.DataFrame.from_records(records, columns=['A', 'B', 'ssim_distance'])
    if opts.csv:
        df.to_csv("/dev/stdout", index=False)
    else:
        print(df.to_string())


