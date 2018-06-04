library(dplyr)
library(stringr)
library(parsetR)
library(webshot)
library(htmltools)
library(magrittr)
library(vcd)
library(entropy)
library(ggplot2)

# Need to install Rafael's version of webshot!
# devtools::install_github("rafaveguim/webshot")

parallelset <- function(..., freq, col="gray", border=0, layer, 
                        alpha=0.5, gap.width=0.05) {
  p <- data.frame(..., freq, col, border, alpha, stringsAsFactors=FALSE)
  n <- nrow(p)
  if(missing(layer)) { layer <- 1:n }
  p$layer <- layer
  np <- ncol(p) - 5
  d <- p[ , 1:np, drop=FALSE]
  p <- p[ , -c(1:np), drop=FALSE]
  p$freq <- with(p, freq/sum(freq))
  col <- col2rgb(p$col, alpha=TRUE)
  if(!identical(alpha, FALSE)) { col["alpha", ] <- p$alpha*256 }
  p$col <- apply(col, 2, function(x) do.call(rgb, c(as.list(x), maxColorValue = 256)))
  getp <- function(i, d, f, w=gap.width) {
    a <- c(i, (1:ncol(d))[-i])
    o <- do.call(order, d[a])
    x <- c(0, cumsum(f[o])) * (1-w)
    x <- cbind(x[-length(x)], x[-1])
    gap <- cumsum( c(0L, diff(as.numeric(d[o,i])) != 0) )
    gap <- gap / max(gap) * w
    (x + gap)[order(o),]
  }
  dd <- lapply(seq_along(d), getp, d=d, f=p$freq)
  par(mar = c(0, 0, 2, 0) + 0.1, xpd=TRUE )
  plot(NULL, type="n",xlim=c(0, 1), ylim=c(np, 1),
       xaxt="n", yaxt="n", xaxs="i", yaxs="i", xlab='', ylab='', frame=FALSE)
  for(i in rev(order(p$layer)) ) {
    for(j in 1:(np-1) )
      polygon(c(dd[[j]][i,], rev(dd[[j+1]][i,])), c(j, j, j+1, j+1),
              col=p$col[i], border=p$border[i])
  }
  text(0, seq_along(dd), labels=names(d), adj=c(0,-2), font=2)
  for(j in seq_along(dd)) {
    ax <- lapply(split(dd[[j]], d[,j]), range)
    for(k in seq_along(ax)) {
      lines(ax[[k]], c(j, j))
      text(ax[[k]][1], j, labels=names(ax)[k], adj=c(0, -0.25))
    }
  }           
}

# data(Titanic)
# myt <- subset(as.data.frame(Titanic), Age=="Adult", 
#               select=c("Survived","Sex","Class","Freq"))
# myt <- within(myt, {
#   Survived <- factor(Survived, levels=c("Yes","No"))
#   levels(Class) <- c(paste(c("First", "Second", "Third"), "Class"), "Crew")
#   color <- ifelse(Survived=="Yes","#008888","#330066")
# })
# 
# with(myt, parallelset(Survived, Sex, Class, freq=Freq, col=color, alpha=0.2))

cbPalette <- c("#999999", "#E69F00", 
               "#56B4E9", "#009E73", 
               "#F0E442", "#0072B2", 
               "#D55E00", "#CC79A7")

simu = function(N_dims, N_levels_mean, N_levels_sd){
  
  # sample number of levels for each dimension
  # from the same probability distribution
  N_levels = pmax(round(rnorm(N_dims, N_levels_mean, N_levels_sd)), 2)
  
  df = expand.grid(lapply(N_levels, function(N) seq(N)))
  
  df %<>%
    mutate(Var1_p = rep(rmultinom(1, 20, rep(1, N_levels[1]))/20, # sample weights for levels in Var1
                        prod(tail(N_levels, -1)))) %>%
    group_by(Var1) %>%
    mutate(Var2_p = rep(rmultinom(1, 20, rep(1/N_levels[2], N_levels[2]))/20,
                        prod(tail(N_levels, -2)))) %>%
    group_by(Var1, Var2) %>%
    mutate(Var3_p = rmultinom(1, 20, rep(1/N_levels[3], N_levels[3]))/20) %>%
    mutate(p = Var1_p*Var2_p*Var3_p) %>%
    ungroup()
}

parset2image = function(df, filepath, 
                        dimensions=NULL,
                        value="p"){
  
  if (is.null(dimensions))
    dimensions = colnames(select(df, matches('Var\\d$')))
  
  value_expression = sprintf("function(d){return d.%s}", value)
  
  # copied from
  html_print(
    parset(df,
       dimensions = dimensions,
       value = htmlwidgets::JS(value_expression),
       width = 640,
       height = 480
    )
  ) %>%
    normalizePath(.,winslash="/") %>%
    gsub(x=.,pattern = ":/",replacement="://") %>%
    paste0("file:///",.) %>%
    webshot(file = filepath, selector='svg')
}

uglyparset2jpeg = function(df, filename){
  res = 96
  width = 640
  height = 480
  jpeg(filename, width, height, res=res)
  with(df, parallelset(Var1, Var2, Var3, freq=p, col=cbPalette[Var1], alpha=0.2))
  dev.off()
}

# - Simulate Data ----------------

# for (i in seq(10)){
#   N_dims = 3
#   N_levels_mean = 2
#   N_levels_sd = 4
#   
#   df = simu(N_dims, N_levels_mean, N_levels_sd)
#   filename = sprintf("data/parset-%d-%d-%d_%d.jpg", N_dims, N_levels_mean, N_levels_sd, i)
#   # uglyparset2jpeg(df, filename) # works for 3 vars only, and it's ugly
#   parset2image(df, filename)
# }

# the uniform distribution chart, this could be the baseline
# parset(df,     
#        colnames(select(df, matches('Var\\d$'))))

# - A Titanic-themed random walk -----

Titanic = as.data.frame(Titanic)

nwalks = 2
nsteps = 20
nvars  = nrow(Titanic)
random_walk = function(nvars, nsteps, mean=1, sd=2){
  # a walk over the integers with step 1 or -1
  # steps  = replicate(nvars, rnorm(nsteps, mean, sd))
  prob  = function() {p1 = runif(1); c(p1, 1-p1)}
  steps = replicate(nvars, sample(c(1,-1), nsteps, replace=TRUE, prob=prob()))
  steps = ifelse(steps>0, 1, -1)
  return(t(steps)) # columns are steps, rows are vars
}

walks = replicate(nwalks,         # a list of matrices
  random_walk(nvars, nsteps), 
  simplify=F) 

# datasets = vector("list", nsteps + 1)
# datasets[[1]] = RandomTitanic

Distance = data.frame(
    walk = rep(seq(nwalks), each=nsteps),
    step = rep(seq(nsteps), nwalks),
    kl = rep(NA, nsteps*nwalks), 
    js = rep(NA, nsteps*nwalks)
)


for (w in seq(nwalks)){
  # start from uniformly distributed data
  RandomTitanic = Titanic
  RandomTitanic %<>% mutate(Freq = rep(10, n()))
  BaselineData = RandomTitanic
  
  for (i in seq(nsteps)){
    walk = walks[[w]]
    RandomTitanic %<>% 
      mutate(Freq = Freq + walk[,i]) %>%
      mutate(Freq = ifelse(Freq < 0, 0, Freq)) # don't let Freq go under 0!
    
    # datasets[[i+1]] = RandomTitanic
    
    # compute data distance measures
    
    m = (BaselineData$Freq + RandomTitanic$Freq)/2 
    kl_pm = KL.empirical(BaselineData$Freq, m)
    kl_qm = KL.empirical(RandomTitanic$Freq, m)
    js = kl_pm/2 + kl_qm/2 # jensen-shannon divergence
    kl = KL.empirical(BaselineData$Freq, RandomTitanic$Freq)
    
    print(sprintf("%d, %d, %f, %f", w, i, kl, js))
    
    r = (w-1)*nsteps + i
    Distance[r, "kl"] = kl
    Distance[r, "js"] = js
    
    parset2image(RandomTitanic,
       sprintf("data/parset-titanic-%d-%d.jpg", w, i),
       dimensions = c("Survived", "Sex", "Age", "Class"),
       value = "Freq")
    
    res = 96
    width = 640
    height = 480
    jpeg(sprintf("data/mosaic-titanic-%d-%d.jpg", w, i), 
         width, height, res=res)
    mosaic(Survived ~ Sex + Age + Class, RandomTitanic)
    dev.off()
  }
}

ggplot(Distance, aes(step, js, group=walk)) +
  geom_point()
