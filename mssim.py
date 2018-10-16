#!/usr/bin/python
#
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Python implementation of MS-SSIM.
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
from scipy import signal
from scipy.ndimage.filters import convolve
from skimage.transform import resize
from pathlib import Path

import imageio
import itertools
import argparse
import cv2
from tqdm import tqdm

from multiprocessing.pool import Pool

from math import factorial
import math

image_cache = dict() # image path -> array[0:M], where M is the number of scales


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


def _SSIMForMultiScale(img1, img2, max_val=255, filter_size=11,
                       filter_sigma=1.5, k1=0.01, k2=0.03, debug=False):
  """Return the Structural Similarity Map between `img1` and `img2`.

  This function attempts to match the functionality of ssim_index_new.m by
  Zhou Wang: http://www.cns.nyu.edu/~lcv/ssim/msssim.zip

  Arguments:
    img1, img2: Numpy array holding an RGB image batch of shape (1, width, height, 3)
          or a grayscale image of shape (1, width, height, 1).
    max_val: the dynamic range of the images (i.e., the difference between the
      maximum the and minimum allowed values).
    filter_size: Size of blur kernel to use (will be reduced for small images).
    filter_sigma: Standard deviation for Gaussian blur kernel (will be reduced
      for small images).
    k1: Constant used to maintain stability in the SSIM calculation (0.01 in
      the original paper).
    k2: Constant used to maintain stability in the SSIM calculation (0.03 in
      the original paper).

  Returns:
    Pair containing the mean SSIM, and the product of the constrast and structure
     components of the SSIM calculation.

  Raises:
    RuntimeError: If input images don't have the same shape or don't have four
      dimensions: [batch_size, height, width, depth].
  """
  if img1.shape != img2.shape:
    raise RuntimeError('Input images must have the same shape (%s vs. %s).',
                       img1.shape, img2.shape)
  if img1.ndim != 4:
    raise RuntimeError('Input images must have four dimensions, not %d',
                       img1.ndim)

  img1 = img1.astype(np.float64)
  img2 = img2.astype(np.float64)

  if debug:
    import matplotlib.pyplot as plt
    plt.imshow(img1[0, :, :, 0], cmap='gray')
    plt.waitforbuttonpress()
    plt.imshow(img2[0, :, :, 0], cmap='gray')
    plt.waitforbuttonpress()

  _, height, width, _ = img1.shape

  # Filter size can't be larger than height or width of images.
  size = min(filter_size, height, width)
  size = max(filter_size, 1)

  # Scale down sigma if a smaller filter size is used.
  sigma = size * filter_sigma / filter_size if filter_size else 0

  # this uses a magic standard dev calculation:
  # see 1) http://matlabtricks.com/post-19/calculating-standard-deviation-using-minimal-memory
  #     2) http://matlabtricks.com/post-20/calculate-standard-deviation-case-of-sliding-window

  if filter_size > 1:
    window = np.reshape(_FSpecialGauss(size, sigma), (1, size, size, 1))
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
  ssim_map = ((2.0 * mu12 + c1) * v1) / ((mu11 + mu22 + c1) * v2)
  ssim = np.mean(ssim_map)
  cs_map = v1 / v2
  cs = np.mean(cs_map)

  if debug:
      plt.imshow(cs_map[0, :, :, 0])
      plt.waitforbuttonpress()

  return ssim, cs


def MultiScaleSSIM(img1_list, img2_list, max_val=255, filter_size=11,
    filter_sigma=1.5, k1=0.01, k2=0.03, weights=None, components=False, debug=False):
  """Return the MS-SSIM score between `img1` and `img2`.

  This function implements Multi-Scale Structural Similarity (MS-SSIM) Image
  Quality Assessment according to Zhou Wang's paper, "Multi-scale structural
  similarity for image quality assessment" (2003).
  Link: https://ece.uwaterloo.ca/~z70wang/publications/msssim.pdf

  Author's MATLAB implementation:
  http://www.cns.nyu.edu/~lcv/ssim/msssim.zip

  Arguments:
    img1_list, img2_list: list holding M images, where M is the number of scales.
        Each element is a Numpy array holding an RGB image batch of shape
        (1, width, height, 3) or a grayscale image of shape (1, width, height, 1).
    max_val: the dynamic range of the images (i.e., the difference between the
      maximum the and minimum allowed values).
    filter_size: Size of blur kernel to use (will be reduced for small images).
    filter_sigma: Standard deviation for Gaussian blur kernel (will be reduced
      for small images).
    k1: Constant used to maintain stability in the SSIM calculation (0.01 in
      the original paper).
    k2: Constant used to maintain stability in the SSIM calculation (0.03 in
      the original paper).
    weights: List of weights for each level; if none, use five levels and the
      weights from the original paper.
    components: If True, return the per-scale SSIM values.

  Returns:
    MS-SSIM score between `img1` and `img2`.

  Raises:
    RuntimeError: If input images don't have the same shape or don't have four
      dimensions: [batch_size, height, width, depth].
  """

  # Note: default weights don't sum to 1.0 but do match the paper / matlab code.
  weights = np.array(weights if weights else
                     [0.0448, 0.2856, 0.3001, 0.2363, 0.1333])

  if isinstance(filter_size, int):
    filter_size = [filter_size]*len(weights)
  if len(filter_size) == 1:
    filter_size = filter_size*len(weights)

  mssim = np.array([])
  mcs = np.array([]) #
  for i, (im1, im2) in enumerate(zip(img1_list, img2_list)):
    ssim, cs = _SSIMForMultiScale(
        im1, im2, max_val=max_val, filter_size=filter_size[i],
        filter_sigma=filter_sigma, k1=k1, k2=k2, debug=debug)
    mssim = np.append(mssim, ssim)
    mcs = np.append(mcs, cs)

  if components:
  	return mcs, mssim
  else:
  	levels = weights.size
  	return (np.prod(mcs[0:levels-1] ** weights[0:levels-1]) *
          (mssim[levels-1] ** weights[levels-1]))


class MSSIM_Caller():
  def __init__(self, **kwargs):
    self.kwargs = kwargs

  def call(self, in_tuple):
    image1_name, image2_name = in_tuple
    img1 = image_cache[image1_name]
    img2 = image_cache[image2_name]
    return (image1_name, image2_name, MultiScaleSSIM(img1, img2, **self.kwargs))


def options():
    parser = argparse.ArgumentParser()
    parser.add_argument('images', nargs='+')
    parser.add_argument('--num_processes', default=2)
    parser.add_argument('--weights', nargs='+',
    	type=float, default=[0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
    parser.add_argument('--components', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--filter_size', nargs='+',
        type=int, default=[11, 9, 7, 5, 3])

    return parser.parse_args()


def main():
  opts = options()

  debug = opts.debug
  num_processes = opts.num_processes
  image_paths = opts.images
  filter_size = opts.filter_size

  images = []
  image_names = [Path(p).stem for p in image_paths]

  for img_path in image_paths:
    HSV = False

    if HSV:
      img = imageio.imread(img_path, format="PNG-PIL")
      img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
      img = np.array([img])
      img = img.astype(np.float64)
      # interpolate hue to 255 (in cv2, hue lies in [0,179])
      img[:, :, :, 0] *= 255/179
    else:
      img = imageio.imread(img_path, format="PNG-PIL", pilmode="L")
      img = np.array([img])
      img = np.expand_dims(img, axis=3)

    print(img.shape)
    images.append(img)

  # Scale all images to lowest size
  width  = min(map(lambda x: x.shape[2], images))
  height = min(map(lambda x: x.shape[1], images))

  for i, img in enumerate(images):
    # img = resize(img, (1, height, width, 1))
    # print(img)
	# downsample img
    downsample_filter = np.ones((1, 2, 2, 1)) / 4.0
    image_cache[image_names[i]] = [img]

    for _ in range(len(opts.weights)-1):
      x = convolve(img, downsample_filter, mode='reflect')
      img = x[:, ::2, ::2, :]
      image_cache[image_names[i]].append(img)

  n_images = len(images)
  n_combinations = factorial(n_images) / (factorial(2) * factorial(n_images-2) )
  pairs = itertools.combinations(image_names, 2)
  pool = Pool(num_processes)
  caller = MSSIM_Caller(weights=opts.weights, components=opts.components, debug=debug, filter_size=filter_size)
  results = []


  with tqdm(total=n_combinations) as pbar:
    for record in tqdm(pool.imap_unordered(caller.call, list(pairs), chunksize=50)):
      results.append(record)
      pbar.update()

  for result in results:
    if opts.components:

      img1_name, img2_name, (mcs, mssim) = result
      print("{},{},{},{}".format(
	    img1_name,
	    img2_name,
	    ','.join(map(str,mcs)),
	    ','.join(map(str,mssim))))
    else:

      img1_name, img2_name, mssim = result
      print("{},{},{:f}".format(
	    img1_name,
	    img2_name,
	    mssim
	  ))



if __name__ == '__main__':
  main()
  # tf.app.run()
