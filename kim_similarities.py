"""
Scans /ouput/kim for groups of images and computes the MS-SSIM
similarity matrix for them.
Outputs the results in files named ssim.csv.
"""

import glob
import os
from pathlib import Path
import imageio
import pandas as pd
import numpy as np
from math import factorial
import itertools
from multiprocessing.pool import Pool
from scipy.ndimage.filters import convolve
from mssim import MultiScaleSSIM

NUM_PROCESSES = 31
MSSIM_WEIGHTS = [0.32, 0.73, 0.82, 1, 1]



class Plot:
    def __init__(self, vars, datasize, stratum, encoding, path):
        self.vars = vars
        self.datasize = datasize
        self.stratum = stratum
        self.encoding = encoding
        self.path = path
        self.name = Path(path).name
        self.image = None
        self.scales = []

        self.load_image()
        self.extract_scales()

    def load_image(self):
        # load image in grayscale mode
        self.image = np.array([imageio.imread(self.path, format="PNG-PIL", pilmode="L")])
        self.image = np.expand_dims(self.image, axis=3)

    def extract_scales(self):
    	# downsample img
        downsample_filter = np.ones((1, 2, 2, 1)) / 4.0
        self.scales.append(self.image)

        img = self.image

        for _ in range(len(MSSIM_WEIGHTS)-1):
          x = convolve(img, downsample_filter, mode='reflect')
          img = x[:, ::2, ::2, :]
          self.scales.append(img)


class MSSIM_Caller():
  def __init__(self, **kwargs):
    self.kwargs = kwargs

  def call(self, in_tuple):
    plot1, plot2 = in_tuple
    return (plot1, plot2, MultiScaleSSIM(plot1.scales, plot2.scales, **self.kwargs))



def calc_sim(folder):
#    if os.path.isfile(folder + "/ssim.csv"):
#        print("Skipping comparisons on folder " + folder)
#        return

    predicates = folder.split('/')
    encoding = predicates[-1]
    stratum  = predicates[-2]
    datasize = predicates[-3]
    vars     = predicates[-4]

    plot_paths = glob.glob(folder + "/*.png")

    plots = []
    for path in plot_paths:
        plots.append(Plot(vars, datasize, stratum, encoding, path))

    n_images = len(plots)
    n_combinations = factorial(n_images) / (factorial(2) * factorial(n_images-2) )
    pairs = list(itertools.combinations(plots, 2))


    fields  = []
    for plot1, plot2 in pairs:
        mcs, mssim = MultiScaleSSIM(plot1.scales, plot2.scales, components=True)
        fields.append((
            folder,
            vars,
            datasize,
            stratum,
            encoding,
            plot1.name,
            plot2.name,
            *mcs,
            *mssim
        ))
    df = pd.DataFrame.from_records(fields, columns=[
        "path", "vars", "datasize", "stratum", "encoding",
        "A", "B", "mcs_1", "mcs_2", "mcs_3", "mcs_4",
        "mcs_5", "mssim_1", "mssim_2", "mssim_3", "mssim_4",
        "mssim_5"
    ])
    df.to_csv(folder + "/ssim.csv")

    print("Finished comparison on folder " + folder)



pool = Pool(NUM_PROCESSES)


folders = glob.glob("output/kim/*/*/*/*")

list(pool.imap_unordered(calc_sim, folders))
