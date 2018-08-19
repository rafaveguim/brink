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
library(dendextend)


PRINT_IMAGE_GRID = T

# Pandey = read.csv("output/pandey_ssim_similarity_rotation_invariant.csv", stringsAsFactors = F)
Pandey = read.csv("output/pandey_ssim_similarity.csv", stringsAsFactors = F)

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

clusters = hclust(as.dist(SimMatrix), method="ward.D")
# clusters = hclust(as.dist(SimMatrix), method="complete")
plot(clusters, labels=F)
summary(clusters)

dend = as.dendrogram(clusters)
dend = color_branches(dend, k=20)
plot(dend, leaflab="none")

sort(tabulate(cutree(dend, k=20)))

# Dynamic Tree Cut

# dynamicTreeCut::cutreeDynamic(clusters, distM = SimMatrix, minClusterSize = 5)
labels = dynamicTreeCut::cutreeDynamicTree(clusters, minModuleSize = 5)

sort(tabulate(labels))

Scatterplots = data.frame(
  plot=colnames(SimMatrix),
  label=labels,
  order=sort(clusters$order, index.return=T)$ix
) 

# plyr::d_ply(Scatterplots,
#   "label",
#   function(subdf){
#     label = subdf$label[1]
#     plot.new()
#     par(mar=rep(0,4)) # no margins
#     N = nrow(subdf)
#     layout(matrix(seq(1, 20), ncol=10, nrow=2, byrow=TRUE))  
#     image_names = subdf$plot
#     for (i in 1:N){
#       img = readPNG(paste0("data/pandey/",image_names[i]))
#       plot(NA,xlim=0:1,ylim=0:1,xaxt="n",yaxt="n",bty="n")
#       rasterImage(img,0,0,1,1)
#     }
#   })

# Arrange image thumbnails according to clustering order
if (PRINT_IMAGE_GRID){
  image_names = colnames(SimMatrix)
  image_names = image_names[clusters$order]
  image_labels = arrange(Scatterplots, order)$label
  N = length(image_names)
  par(mar=rep(0,4)) # no margins
  
  # layout the plots into a matrix w/ 19 columns, by row
  layout(matrix(seq(1, N), ncol=19, byrow=TRUE))
  
  for (i in 1:N){
    lab_prev = if (i==1) 1000 else image_labels[i-1]
    lab = image_labels[i]
    
    img = readPNG(paste0("data/pandey/",image_names[i]))
    plot(NA,xlim=0:1,ylim=0:1,xaxt="n",yaxt="n",bty="n")
    
    rasterImage(img,0,0,1,1)
    if (lab_prev != lab)
      text(0.2,0.2,lab,col="red", cex=1.5)
  }
}
