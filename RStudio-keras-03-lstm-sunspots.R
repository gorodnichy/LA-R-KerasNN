# RStudio-keras-03-lstm.R
# Deep Learning for Time Series Forecasting: Predicting Sunspot Frequency with Keras
# From: https://tensorflow.rstudio.com/blog/sunspots-lstm.html
#


#.################################################################################


#' Use Convolution1D for text classification.
#'
#' Output after 2 epochs: ~0.89
#' Time per epoch on CPU (Intel i5 2.4Ghz): 90s
#' Time per epoch on GPU (Tesla K40): 10s
#'

# Core Tidyverse
library(tidyverse)
library(glue)
#library(forcats)

# Time Series
library(timetk)
library(tidyquant)
library(tibbletime)

# Visualization
library(cowplot)

# Preprocessing
library(recipes)

# Sampling / Accuracy
library(rsample) # rolling_origin()
library(yardstick)

# Modeling
library(keras)
library(tfruns)


# Data ----


sun_spots <- datasets::sunspot.month %>%
  tk_tbl() %>%
  mutate(index = as_date(index)) %>%
  as_tbl_time(index = index)
sun_spots



p1 <- sun_spots %>%
  ggplot(aes(index, value)) +
  geom_point(color = palette_light()[[1]], alpha = 0.5) +
  theme_tq() +
  labs(
    title = "From 1749 to 2013 (Full Data Set)"
  )

p2 <- sun_spots %>%
  filter_time("start" ~ "1800") %>%
  ggplot(aes(index, value)) +
  geom_line(color = palette_light()[[1]], alpha = 0.5) +
  geom_point(color = palette_light()[[1]]) +
  geom_smooth(method = "loess", span = 0.2, se = FALSE) +
  theme_tq() +
  labs(
    title = "1749 to 1759 (Zoomed In To Show Changes over the Year)",
    caption = "datasets::sunspot.month"
  )

p_title <- ggdraw() +
  draw_label("Sunspots", size = 18, fontface = "bold", colour = palette_light()[[1]])

plot_grid(p_title, p1, p2, ncol = 1, rel_heights = c(0.1, 1, 1))



# Developing a backtesting strategy with rsample: ----


periods_train <- 12 * 100
periods_test  <- 12 * 50
skip_span     <- 12 * 22 - 1

rolling_origin_resamples <- rolling_origin(
  sun_spots,
  initial    = periods_train,
  assess     = periods_test,
  cumulative = FALSE,
  skip       = skip_span
)

rolling_origin_resamples
rolling_origin_resamples$splits[1]


# Plotting function for a single split
plot_split <- function(split, expand_y_axis = TRUE, alpha = 1, size = 1, base_size = 14) {

  # Manipulate data
  train_tbl <- training(split) %>%
    add_column(key = "training")

  test_tbl  <- testing(split) %>%
    add_column(key = "testing")

  data_manipulated <- bind_rows(train_tbl, test_tbl) %>%
    as_tbl_time(index = index) %>%
    mutate(key = fct_relevel(key, "training", "testing"))

  # Collect attributes
  train_time_summary <- train_tbl %>%
    tk_index() %>%
    tk_get_timeseries_summary()

  test_time_summary <- test_tbl %>%
    tk_index() %>%
    tk_get_timeseries_summary()

  # Visualize
  g <- data_manipulated %>%
    ggplot(aes(x = index, y = value, color = key)) +
    geom_line(size = size, alpha = alpha) +
    theme_tq(base_size = base_size) +
    scale_color_tq() +
    labs(
      title    = glue("Split: {split$id}"),
      subtitle = glue("{train_time_summary$start} to {test_time_summary$end}"),
      y = "", x = ""
    ) +
    theme(legend.position = "none")

  if (expand_y_axis) {

    sun_spots_time_summary <- sun_spots %>%
      tk_index() %>%
      tk_get_timeseries_summary()

    g <- g +
      scale_x_date(limits = c(sun_spots_time_summary$start,
                              sun_spots_time_summary$end))
  }

  return(g)
}



rolling_origin_resamples$splits[[1]] %>%
  plot_split(expand_y_axis = TRUE) +
  theme(legend.position = "bottom")



# Plotting function that scales to all splits
plot_sampling_plan <- function(sampling_tbl, expand_y_axis = TRUE,
                               ncol = 3, alpha = 1, size = 1, base_size = 14,
                               title = "Sampling Plan") {

  # Map plot_split() to sampling_tbl
  sampling_tbl_with_plots <- sampling_tbl %>%
    mutate(gg_plots = map(splits, plot_split,
                          expand_y_axis = expand_y_axis,
                          alpha = alpha, base_size = base_size))

  # Make plots with cowplot
  plot_list <- sampling_tbl_with_plots$gg_plots

  p_temp <- plot_list[[1]] + theme(legend.position = "bottom")
  legend <- get_legend(p_temp)

  p_body  <- plot_grid(plotlist = plot_list, ncol = ncol)

  p_title <- ggdraw() +
    draw_label(title, size = 14, fontface = "bold", colour = palette_light()[[1]])

  g <- plot_grid(p_title, p_body, legend, ncol = 1, rel_heights = c(0.05, 1, 0.05))

  return(g)

}


rolling_origin_resamples %>%
  plot_sampling_plan(expand_y_axis = T, ncol = 3, alpha = 1, size = 1, base_size = 10,
                     title = "Backtesting Strategy: Rolling Origin Sampling Plan")

rolling_origin_resamples %>%
  plot_sampling_plan(expand_y_axis = F, ncol = 3, alpha = 1, size = 1, base_size = 10,
                     title = "Backtesting Strategy: Zoomed In")



# The LSTM model ----
#

example_split    <- rolling_origin_resamples$splits[[6]]
example_split_id <- rolling_origin_resamples$id[[6]]

plot_split(example_split, expand_y_axis = FALSE, size = 0.5) +
  theme(legend.position = "bottom") +
  ggtitle(glue("Split: {example_split_id}"))


# Data setup ----
#
df_trn <- analysis(example_split)[1:800, , drop = FALSE]
df_val <- analysis(example_split)[801:1200, , drop = FALSE]
df_tst <- assessment(example_split)


df <- bind_rows(
  df_trn %>% add_column(key = "training"),
  df_val %>% add_column(key = "validation"),
  df_tst %>% add_column(key = "testing")
) %>%
  as_tbl_time(index = index)

df
df_tst <- assessment(example_split)

# Preprocessing with recipes ----


rec_obj <- recipe(value ~ ., df) %>%
  step_sqrt(value) %>% # we’re using step_sqrt to reduce variance and remov outliers.
  step_center(value) %>%
  step_scale(value) %>%
  prep()

df_processed_tbl <- bake(rec_obj, df)

df_processed_tbl


center_history <- rec_obj$steps[[2]]$means["value"]
scale_history  <- rec_obj$steps[[3]]$sds["value"]

c("center" = center_history, "scale" = scale_history)
