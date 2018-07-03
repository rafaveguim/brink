# This script reads a CSV file produced by similarity_matrix.py
# that contains image similarity measures for the
# image dataset used in "Towards Understanding Human Similarity Perception 
# in the Analysis of Large Sets of Scatter Plots".
# It generates a hierarchical clustering of the images
# and outputs an image similar to Figure 5 in the aforementioned
# paper.

library(tidyr)
library(plyr)
library(dplyr)
library(png)

Pandey = read.csv("output/pandey_ssim_similarity_rotation_invariant.csv", stringsAsFactors = F)

# Assume we have observations in a single direction
# Example: a - b, but not b - a. Fill in these gaps.

temp = Pandey
temp$A = Pandey$B
temp$B = Pandey$A
Pandey = rbind(Pandey, temp)

# similarity -> dissimilarity
Pandey = mutate(Pandey, ssim_distance=1-ssim_distance)

# Transform pairwise list into similarity matrix

image_names = unique(c(Pandey$A, Pandey$B))
SimMatrix = expand.grid(A=image_names, B=image_names)
SimMatrix = left_join(SimMatrix, Pandey,)
SimMatrix = spread(SimMatrix, B, ssim_distance)
SimMatrix = data.matrix(SimMatrix, rownames.force = F)[,-1]
SimMatrix[is.na(SimMatrix)] = 0
row.names(SimMatrix) = colnames(SimMatrix)

# MDS

mds = cmdscale(SimMatrix, eig = TRUE, k = 2)
x <- mds$points[, 1]
y <- mds$points[, 2]
plot(x, y)

# Hierarchical clustering

clusters = hclust(as.dist(SimMatrix))
plot(clusters, labels=F)
summary(clusters)

# Arrange image thumbnails according to clustering order

image_names = colnames(SimMatrix)
image_names = image_names[clusters$order]
N = length(image_names)
par(mar=rep(0,4)) # no margins

# layout the plots into a matrix w/ 12 columns, by row
layout(matrix(seq(1, N), ncol=19, byrow=TRUE))

for (i in 1:N){
  img = readPNG(paste0("data/pandey/",image_names[i]))
  plot(NA,xlim=0:1,ylim=0:1,xaxt="n",yaxt="n",bty="n")
  rasterImage(img,0,0,1,1)
}

