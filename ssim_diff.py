import sys
import cv2
import numpy as np
import skimage.measure as measures
import imutils
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
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--outfile')
    return parser.parse_args()

def compare(imageA, imageB, measure=None):
    if not measure:
        measure = measures.compare_ssim
    return measure(imageA, imageB)

def scale(img, width, height):
    return cv2.resize(img, (width, height), interpolation = cv2.INTER_CUBIC)


if __name__ == '__main__':
    opts = options()

    images = []

    for image_path in opts.images:
        image_path = Path(image_path)

        img = cv2.imread(str(image_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        images.append((image_path, img))

    # Scale all images to lowest size
    width  = min(map(lambda x: x[1].shape[1], images))
    height = min(map(lambda x: x[1].shape[0], images))

    images = [(imgpath, scale(img, width, height)) for imgpath, img in tqdm(images)]

    for a, b in tqdm(list(itertools.combinations(images, 2))):
        imgA_path, imgA = a
        imgB_path, imgB = b

        (value, diff) = measures.compare_ssim(imgA, imgB, full=True)

        diff = (diff * 255).astype("uint8")

        thresh = cv2.threshold(diff, 0, 255,
            cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]

        # loop over the contours
        for c in cnts:
            # compute the bounding box of the contour and then draw the
            # bounding box on both input images to represent where the two
            # images differ
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(imgA, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.rectangle(imgB, (x, y), (x + w, y + h), (0, 0, 255), 2)

        if not opts.save:
            # cv2.imshow("Original", imgA)
            # cv2.imshow("Modified", imgB)
            cv2.imshow("Diff", diff)
            # cv2.imshow("Thresh", thresh)
            cv2.waitKey(0)
        else:
            outfile = opts.outfile
            if not outfile:
                outfile = 'diff_' + imgA_path.stem + '_' + imgB_path.name
            cv2.imwrite(outfile, diff)

        # if is_rotation_invariant:
        #     imgB_90, imgB_180, imgB_270 = rotations[imgB_path]

        #     value_90  = compare(imgA, imgB_90,  measure)
        #     value_180 = compare(imgA, imgB_180, measure)
        #     value_270 = compare(imgA, imgB_270, measure)

        #     value = max(value, value_90, value_180, value_270)
