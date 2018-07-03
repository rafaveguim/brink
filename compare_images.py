import sys
import cv2
import numpy as np
import skimage.measure as measures

import argparse
import pandas as pd

from pathlib import Path

def options():
	parser = argparse.ArgumentParser()
	parser.add_argument('imageA')
	parser.add_argument('imageB', nargs='+')
	parser.add_argument('--csv', action='store_true')
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
	


if __name__ == '__main__':
	opts = options()

	imageA_path = Path(opts.imageA)
	original = cv2.imread(str(imageA_path))
	original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

	records = []

	for imageB_path in opts.imageB:
		imageB_path = Path(imageB_path)

		contrast = cv2.imread(str(imageB_path))
		contrast = cv2.cvtColor(contrast, cv2.COLOR_BGR2GRAY)

		ssim_distance  = 1 - compare(original, contrast, measures.compare_ssim)
		nrmse_distance =     compare(original, contrast, measures.compare_nrmse)

		records.append((imageA_path.name, 
			imageB_path.name,
			ssim_distance,
			nrmse_distance))

	df = pd.DataFrame.from_records(records, columns=['A', 'B', 'ssim', 'nrmse'])
	if opts.csv:
		df.to_csv(sys.stdout, index=False)
	else:	
		print(df.to_string())


