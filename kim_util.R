library(jsonlite)
library(ggplot2)
library(dplyr)

CODE_2_STATE <- c(
  "AL" = "Alabama",
  "AK" = "Alaska",
  "AZ" = "Arizona",
  "AR" = "Arkansas",
  "CA" = "California",
  "CO" = "Colorado",
  "CT" = "Connecticut",
  "DE" = "Delaware",
  "FL" = "Florida",
  "GA" = "Georgia",
  "HI" = "Hawaii",
  "ID" = "Idaho",
  "IL" = "Illinois",
  "IN" = "Indiana",
  "IA" = "Iowa",
  "KS" = "Kansas",
  "KY" = "Kentucky",
  "LA" = "Louisiana",
  "ME" = "Maine",
  "MD" = "Maryland",
  "MA" = "Massachusetts",
  "MI" = "Michigan",
  "MN" = "Minnesota",
  "MS" = "Mississippi",
  "MO" = "Missouri",
  "MT" = "Montana",
  "NE" = "Nebraska",
  "NV" = "Nevada",
  "NH" = "New Hampshire",
  "NJ" = "New Jersey",
  "NM" = "New Mexico",
  "NY" = "New York",
  "NC" = "North Carolina",
  "ND" = "North Dakota",
  "OH" = "Ohio",
  "OK" = "Oklahoma",
  "OR" = "Oregon",
  "PA" = "Pennsylvania",
  "RI" = "Rhode Island",
  "SC" = "South Carolina",
  "SD" = "South Dakota",
  "TN" = "Tennessee",
  "TX" = "Texas",
  "UT" = "Utah",
  "VT" = "Vermont",
  "VA" = "Virginia",
  "WA" = "Washington",
  "WV" = "West Virginia",
  "WI" = "Wisconsin",
  "WY" = "Wyoming"
)

STATE_2_CODE <- names(CODE_2_STATE)
names(STATE_2_CODE) <- CODE_2_STATE

simulate.betareg <- function(model, nsim = 1, XX = NULL, ...) {
  list(
    sim_1 = as.vector(posterior_predict(model, newdata = XX, draws = nsim))
  )
}

kim_get_tasks <- function(taskgroup=NULL) {
  df = read.csv("data/kim/responses.csv")
  
  if (!is.null(taskgroup)){
    df = filter(df, taskGroup == taskgroup)
  }

  df = select(df,
    question, assignmentID, correctOption, taskGroup, dataset,
    Q1, Q2, N, stratum, Q1_channel, Q2_channel, name_channel,
    channel, options0, options1, correctOption, cardinality,
    nPerCategory
  )
  distinct(df)
}


kim_datapath <- function(dataset) {
  path <- str_replace(dataset, "(.+[0-9])_(.+)", "\\1/\\2")
  sprintf("data/kim/%s", path)
}

#' Find all observations of a variable.
#'
#' Given two variables, load and consolidate all datasets
#' containing them.
#'
#' @param tasks a data frame containing task metadata
#' @param v variable
#'
#' @return a data frame
#'
get_representative_sample <- function(tasks, v) {
  relevant_datasets <- (tasks %>%
    filter(Q1 == v))$dataset
  relevant_datasets %<>% as.character
  relevant_paths <- str_replace(relevant_datasets, "(.+[0-9])_(.+)", "\\1/\\2")

  relevant_paths <- sprintf("data/kim/%s", relevant_paths)
  adply(relevant_paths, 1, fromJSON)
}


kim_plot <- function(data, encoding) {
  Vars <- str_split(encoding, " ", simplify = T)
  y <- NULL
  x <- NULL
  color <- NULL
  size <- NULL
  row <- NULL
  eval(parse(text = sprintf('%s = "%s"', Vars[1], colnames(data)[1]))) # like x = "TMAX"
  eval(parse(text = sprintf('%s = "%s"', Vars[2], colnames(data)[2])))
  eval(parse(text = sprintf('%s = "%s"', Vars[3], colnames(data)[3])))

  question_var <- colnames(data)[1]

  data$name <- factor(data$name)
  
  data %>%
    ggplot(aes_string(x = x, y = y, color = color, size = size)) +
    geom_point(shape = 21) +
    # do.call(lims, limits) +
    theme_minimal() +
    theme(
      legend.position = "none",
      panel.grid = element_blank(),
      axis.text.y = element_blank(),
      axis.text.x = element_blank()
    ) -> gg_data
  
  if (!is.null(row)) {
    gg_data <- gg_data +
      facet_wrap(as.formula(paste("~", row)),
        ncol = 1
      )
  }

  return(gg_data)
}

kim_responses <- function(){
  
}


# df = read.csv("data/kim/responses.csv")


