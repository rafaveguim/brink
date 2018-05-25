library(jpeg)
library(colordistance)

filepath = 'data/3-2-4_1.jpg'
img = readJPEG(filepath)
imgsize = file.size(filepath)

# colorpal <- c("#999999", "#E69F00",  # the color palette:
#               "#56B4E9", "#009E73",  # a pixel is expected to be
#               "#F0E442", "#0072B2",  # one of these colors
#               "#D55E00", "#CC79A7",
#               "#ffffff", "#000000")  # black and white

# do.call(rgb, as.list(img[200,100,1:4]))

# defaultHist = getImageHist('data/3-4-2.png', 
#   bins=8, upper=NULL, lower=NULL, hsv=TRUE)

# plotPixels('data/3-4-2.png', lower=NULL, upper=NULL, hsv=TRUE)


# TODO:
# * partition figure and compute compression difference (NCD)
# * search for standard image similarity measures
# * look for Kosara's parallel sets (is it open source?)