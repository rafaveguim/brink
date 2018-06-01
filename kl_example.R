set.seed(3951824)
# Table of counts (contingency)
X <- sample(letters[1:3], 100, 1)
Y <- sample(letters[4:5], 100, 1)
t  <- table(X,Y)
tp  <- t/100 # proportions
tn  <- tp/sum(tp)     # normalized, joints
p_x <- rowSums(tn)    # marginals
p_y <- colSums(tn)

P <- tn 
Q <- p_x %o% p_y 

# P(X, Y)   : bin frequency: P_i - joint distribution 
# P(X) P(Y) : bin frequency, Q_i - product of marginal distributions
mi <- sum(P*log(P/Q))
library(entropy)
mi.empirical(t) == mi   # mutual information is the KL divergence between
                        # P(X,Y) and P(X)P(Y)

# Notes
# the joint distribution captures the dependency between variables
# product of marginal distributions is the probability of outcomes if variables were independent
# mutual information is the difference between the two
