#' This experiment seeks to verify if discriminability can
#' explain the results of Kim and Heer (2018).
#' One reference dataset will be selected for each stratum.
#' A stratum is a tuple of the factor levels used in the study:
#' (entropy_q1, entropy_q2, entropy_n, clusteredness, correlation)
#' Multiple derivations will be produced from this reference
#' by simulating outcomes for the "question variable".


library(checkpoint)

checkpoint("2018-08-01", scanForPackages = F)

library(MASS)
library(plyr)
library(dplyr)
library(magrittr)
library(stringr)
library(memoise)
library(arm)
source("kim_util.R")

library(parallel)

select <- dplyr::select

fitnullmodel <- function(data, v){
  
  if (v == "PRCP"){
    data %<>% mutate(PRCP=ifelse(PRCP <= 0, 0.001, PRCP))
    null_model = glm(PRCP ~ 1, data=data, family=Gamma)
  } 
  else if (v == "AWND"){
    data %<>% mutate(AWND = ifelse(AWND == 0, 0.00001, AWND))
    null_model <- glm(AWND ~ name, data=data, family=Gamma)
  } 
  else if (v == "WSF5"){
    data %<>% mutate(WSF5 = ifelse(WSF5 == 0, 0.00001, WSF5))
    null_model = glm(WSF5 ~ name, data=data, family=Gamma)
  } 
  else if (v == "SNOW"){
    data %<>% mutate(SNOW = ifelse(SNOW == 0, 0.000001, SNOW))
    null_model = glm(SNOW ~ name, data=data, family=Gamma)
  }
  else {
    f = as.formula(sprintf("%s ~ name", v))
    null_model = do.call(lm, list(f, data=data))
  }
  
  return(null_model)
}

simulate <- function(model, data) {
  if (family(model)$family == "Gamma") {
    # https://stats.stackexchange.com/a/109188/74147
    # https://stats.stackexchange.com/a/176531/74147
    shape <- gamma.shape(model)$alpha
    mu <- predict(model, type = "response", newdata = data)
    scale <- mu / shape
    n_ <- length(scale)
    sim <- mapply(rgamma, rep(1, n_), rep(shape, n_), scale = scale)
  } else if (family(model)$family == "beta") {
    # https://stats.stackexchange.com/q/12232/74147
    phi <- coef(model)["(phi)"]
    mu <- predict(model, newdata = data, type = "response")
    alpha <- phi * mu
    beta_ <- phi - alpha
    n_ <- length(mu)
    sim <- mapply(rbeta, rep(1, n_), alpha, beta_)
  } else {
    stdev <- sigma(model)
    mu <- predict(model, type = "response", newdata = data)
    n_ <- nrow(data)
    sim <- mapply(rnorm, rep(1, n_), mu, rep(stdev, n_))
  }
}

get_representative_sample <- memoise(get_representative_sample) #### <---- Memoise

N_bootstrap = 20 # the number of synthetic datasets to produce for each stratum

Entropy_q1   <- c("H", "L")
Entropy_q2   <- c("H", "L")
Entropy_n    <- c("H")

# Interestingly, the levels in the paper do not match the data levels
# Paper levels:
# Cardinality -> 3, 10, 30
# N_per_category -> 1, 10
# =O =O =O
Cardinality  <- c(3, 10, 20)
N_per_category <- c(3, 30)
Variables    <- c("TMAX", "TMIN", "AWND", "SNOW", "SNWD", "PRCP", "WSF5", "WDF5")
Encodings    <- c("x y color",
  "x y row",
  "size x y",
  "x size y",
  "color x y",
  "y x row",
  "y x color",
  "y size x",
  "size y x",
  "y color x",
  "x color y",
  "color y x")

Strata <- expand.grid(
  entropy_q1   = Entropy_q1,
  entropy_q2   = Entropy_q2,
  entropy_n    = Entropy_n,
  cardinality  = Cardinality,
  n_per_category = N_per_category,
  stringsAsFactors = F
)

Vars <- expand.grid(q1 = Variables, q2 = Variables, stringsAsFactors = F)
# Q1 cannot be equal to Q2 or refer to the same quantity (e.g., TMAX and TMIN)
Vars %<>% filter(q1!=q2) %>% filter(str_sub(q1,1,1) != str_sub(q2,1,1))

stratumcode <- function(a) {
  clusteredness <- ifelse(a$entropy_q1 == "L", ifelse(a$entropy_q2 == "L", "L", "H"), "H")
  
  sprintf(
    "%d_%d/%s%s%s_%s__W",
    a$cardinality,
    a$n_per_category,
    a$entropy_q1,
    a$entropy_q2,
    a$entropy_n,
    clusteredness
  )
}

KimTasks = kim_get_tasks("compareAggregatedValue")
KimTasks %<>% mutate_if(is.factor, as.character)

set.seed(123)


make_plots <- function(q1, q2){
  
  for (i in 1:nrow(Strata)) {
    
    # let's sample one dataset from this stratum
    stratum <- Strata[i, ]
    stratum_code <- stratumcode(stratum)
    stimulus_meta <- (KimTasks 
      %>% filter(Q1 == q1, Q2 == q2)
      %>% filter(assignmentID == stratum_code))
    
    if (nrow(stimulus_meta) == 0){
      cat(sprintf("Skipping %s_%s %s\n", q1, q2, stratum_code))
      next
    }
    
    stimulus_meta %<>% sample_n(1)
    
    
    # so we need to load the data and find the relevant state-months
    stimulus_data <- fromJSON(kim_datapath(stimulus_meta$dataset))
    stimulus_data <- stimulus_data[, c(q1, q2, "name")]
    states <- unique(stimulus_data$name)
    
    # now gather all known observations of the variable q1
    # with the purpose of fitting a GLM from which we will 
    # sample new values
    bigdata <- get_representative_sample(KimTasks, q1)
    bigdata %<>% filter(name %in% states)
    
    nullmodel <- fitnullmodel(bigdata, q1)
    
    for (k in 1:N_bootstrap){
      nulldata <- select(stimulus_data, -one_of(c(q1)))
      nulldata[, q1] <- simulate(nullmodel, nulldata)
      nulldata <- nulldata[, c(q1, q2, "name")]
      
      for(e in 1:length(Encodings)){
        encoding <- Encodings[e]
        nullplot <- kim_plot(nulldata, encoding)
        
        output_folder = sprintf("output/kim/%s/%s/%s",
          paste(q1, q2, sep="_"),
          stratum_code,
          str_replace_all(encoding, " ", "_"))
        
        if (!dir.exists(output_folder)){
          dir.create(output_folder, recursive = T)
        } 
        
        output_path = sprintf("%s/%02d.png", output_folder, k)
        
        ncat = stimulus_meta$cardinality
        
        # if name is x
        dpi = 96
        width = if (str_detect(encoding, "x$")) 1 + ncat*.15 else 3 
        
        height = if (str_detect(encoding, "row$")) 1 + ncat*3
        else if (str_detect(encoding, "y$")) 1 + ncat*.15
        else 3
        
        ggsave(output_path, 
          nullplot, 
          width = width, 
          height = height, 
          dpi = dpi, 
          limitsize = F)
      }
    }
  }
}



# ----------- Core Computation Here ---------------

ncores = detectCores()

mclapply(1:nrow(Vars), function(i){
  q1 = Vars[i, "q1"]
  q2 = Vars[i, "q2"]
  # 
  make_plots(q1, q2)
  print(sprintf("Finished folder %s_%s", q1, q2))  
}, mc.cores=ncores-1)
