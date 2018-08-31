# This script reads a CSV file produced by similarity_matrix.py
# that contains image similarity measures for the
# image dataset used in "Towards Understanding Human Similarity Perception 
# in the Analysis of Large Sets of Scatter Plots".
# It generates a hierarchical clustering of the images
# and outputs an image similar to Figure 5 in the aforementioned
# paper.

library(checkpoint)
checkpoint("2018-05-01", scanForPackages = F)

library(tidyr)
library(plyr)
library(dplyr, warn.conflicts = FALSE)
library(png)
suppressPackageStartupMessages(
  library(dendextend)
)
library(dynamicTreeCut)
library(argparser, quietly = TRUE)


parser = arg_parser("Generate SSIM clusters of the Pandey data")
parser = add_argument(parser, "ssim_file", help="File with SSIM scores (pairwise)")
parser = add_argument(parser, "output", help="File where cluster labels will be ouput")
parser = add_argument(parser, "--no_header", help="Flag indicating the SSIM file has no header", flag=TRUE)
parser = add_argument(parser, "--plot_clustering", help="Filepath where image should be saved")

argv = parse_args(parser)

PRINT_IMAGE_GRID = !is.na(argv$plot_clustering)
SSIM_FILE = argv$ssim_file
OUT_FILE = argv$output
FLAG_NO_HEADER = argv$no_header

if (PRINT_IMAGE_GRID)
  PLOT_FILE = argv$plot_clustering

Pandey = read.csv(SSIM_FILE, 
  stringsAsFactors = F, 
  header = !FLAG_NO_HEADER,
  strip.white = TRUE)

if (FLAG_NO_HEADER)
  colnames(Pandey) = c("A", "B", "measure")


# If we have observations in a single direction,
# Fill in the gaps.
# Example: a - b, but not b - a.

image_names = unique(c(Pandey$A, Pandey$B))
n_images = length(image_names)

if (nrow(Pandey) < n_images*n_images){
  temp = Pandey
  temp$A = Pandey$B
  temp$B = Pandey$A
  Pandey = rbind(Pandey, temp)  
} 


# similarity -> dissimilarity
Pandey = mutate(Pandey, measure=1-measure)

# Transform pairwise list into similarity matrix
SimMatrix = expand.grid(A=image_names, B=image_names)
SimMatrix = left_join(SimMatrix, Pandey)
SimMatrix = spread(SimMatrix, B, measure)
SimMatrix = data.matrix(SimMatrix, rownames.force = F)[,-1]
SimMatrix[is.na(SimMatrix)] = 0
row.names(SimMatrix) = colnames(SimMatrix)

# MDS

mds = cmdscale(SimMatrix, eig = TRUE, k = 2)
x <- mds$points[, 1]
y <- mds$points[, 2]
plot(x, y)

# Hierarchical clustering

clusters = hclust(as.dist(SimMatrix), method="ward.D2")
# clusters = hclust(as.dist(SimMatrix), method="complete")
plot(clusters, labels=F)
# summary(clusters)

dend = as.dendrogram(clusters)
dend = color_branches(
  dend,
  k=19)
plot(dend, leaflab="none")

# Dynamic Tree Cut

# dynamicTreeCut::cutreeDynamic(clusters, distM = SimMatrix, minClusterSize = 5)
labels = dynamicTreeCut::cutreeDynamicTree(clusters, minModuleSize = 8)
# labels = cutree(dend, k=19)

cat("\n# of clusters: ", length(unique(labels)), '\n')
cat("Cluster sizes: ", toString(sort(tabulate(labels))), '\n')

Scatterplots = data.frame(
  plot=colnames(SimMatrix),
  label=labels,
  order=sort(clusters$order, index.return=T)$ix
) 

# Arrange image thumbnails according to clustering order
if (PRINT_IMAGE_GRID){
  png(PLOT_FILE, 1024, 768)
  
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
    
    img = readPNG(paste0("data/pandey/thumbnails/",image_names[i]))
    plot(NA,xlim=0:1,ylim=0:1,xaxt="n",yaxt="n",bty="n")
    
    rasterImage(img,0,0,1,1)
    if (lab_prev != lab)
      text(0.2,0.2,lab,col="red", cex=1.5)
  }
  
  dev.off()
}

# Output clustering
write.csv(Scatterplots, 
  OUT_FILE,
  row.names = F)
