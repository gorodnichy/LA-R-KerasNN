# RStudio-keras-00-ToC+HelloWorld.R (with simplest MNIST digit recognition)

# `Learn and Apply Keras/Tensorflow in R/RStudio Efficiently!`  <<< ####
# Using complete prewritten codes, your own data, and data.table package
#
# Based on:
# - https://github.com/jjallaire/deep-learning-with-r-notebooks
# - https://keras.rstudio.com (= https://tensorflow.rstudio.com/keras)
# - https://www.manning.com/books/deep-learning-with-r
# - everything else found useful on the Web
#   - https://github.com/PacktPublishing/R-Deep-Learning-Cookbook
#   - See also RStudio-tf-*.R
# - and my own work back from 1995-2005 on PINN (www.videorecognition.com/memory/pinn)
#
# Notes:
# - Latest version of RStudio is always recommended (Presently, Version 1.1.447 – 2018 )
# - All codes are retreived from original public sources, modified, directly runnable from RStudio
# - Where possible, `library(data.table)` is used
#
# - The order of lessons is recommended by indices: e.g. 1-1-1 goes prior to 1-3-1. [.] are optional.
# - `# .... ####` comments are used for quick navigation from one example/section to another
# - `# >>> ... <<< ####` indicate Main sections
# - Data to play with (traffic, favourite readings) are provided, inc. very small sets to run fast.
#

# 1. Contents: ####
#
# 1. Start here: https://keras.rstudio.com/index.html (which is the same as  https://tensorflow.rstudio.com/keras)
# Then, as instructed there go to. # Learning More:

# 1-2. Guide to the Sequential Model - https://keras.rstudio.com/articles/sequential_model.html
# Then, as instructed there go to # Examples:
#
#
# [1-2-1]. CIFAR10 small images classification -      https://keras.rstudio.com/articles/examples/cifar10_cnn.html
#
# 1-2-2. IMDB movie review sentiment classification - https://keras.rstudio.com/articles/examples/imdb_cnn_lstm.html
# 1-2-3. Reuters newswires topic classification -     https://keras.rstudio.com/articles/examples/reuters_mlp.html
# 1-2-4. MNIST handwritten digits classification -    https://keras.rstudio.com/articles/examples/mnist_mlp.html
# = DLwR-s3-3ClassicML=IMDB_Binary+wiresClassification+housepriceReression.R
#
# 1-2-5. Predicting Sunspot Frequency - https://tensorflow.rstudio.com/blog/sunspots-lstm.html
#
# 1-2-6. Simple audio classification - https://tensorflow.rstudio.com/blog/simple-audio-classification-keras.html
#
#
# 1-3. Guide to the Functional API -    https://keras.rstudio.com/articles/functional_api.html
# [1-4]. Frequently Asked Questions -   https://keras.rstudio.com/articles/faq.html
# 1-1. Training Visualization -         https://keras.rstudio.com/articles/training_visualization.html

# Other files:
# - DLwR-s6.1-RNN-for-text.R
# - DLwR-s6.1-RNN-forSequences.R
#

# Libraries used ----
library(data.table);library(magrittr)
library(tibble); library(readr); library(magrittr); library(ggplot2); library(dplyr)
library(tidyverse)
library(keras)

################################################################################################ #
################################################################################################ #

# 1-0 "Hello World" for Keras: MNIST 28x28 digit recognition ####
# . = # 1-2-4. 0 > mnist_mlp.R ####
# https://keras.rstudio.com/index.html
# = https://tensorflow.rstudio.com/keras/
#
# Trains a simple deep NN on the MNIST dataset.
# Gets to 98.40% test accuracy after 20 epochs (there is a lot of margin for parameter tuning). 2 seconds per epoch on a K520 GPU.


# Get and prepare mnist data-set ----

if (F) {
  c(c(x_train, y_train), c(x_test, y_test)) %<-% dataset_mnist()
  if (F) {# this the same as:
    mnist <- dataset_mnist();
    x_train <- mnist$train$x; y_train <- mnist$train$y;
    x_test <- mnist$test$x; y_test <- mnist$test$y
  }
  rm(mnist);
}

mnist <- dataset_mnist();
str(mnist$train$x) # int [1:60000, 1:28, 1:28] 0 0 0 0 0 0 0 0 0 0 ...
str(mnist$train$y) # int [1:60000(1d)] 5 0 4 1 9 2 1 3 1 4 ...
str(mnist$test$x)  # int [1:10000, 1:28, 1:28] 0 0 0 0 0 0 0 0 0 0 ...
mnist$train$x[1,,] # - See a digit

# Get data sizes and dimensions:

data_quantity <- dim(mnist$train$x)[1] # 60000 # nrow(dt)
data_dim <- list(); for (i in 2:length(dim(mnist$train$x)))   data_dim[[i-1]] <- dim(mnist$train$x)[i]
# Input image dimensions
img_rows <- data_dim[[1]] #28 #
img_cols <- data_dim[[2]] #28
num_classes <- mnist$train$y %>% unique %>% length() # 10 # 10 digits

# make small subset: take 100 samples of each digit (instead of 6000) for training, and 50 (instead of 1000) for testing

sizeTrain <- 10

dt <- data.table(x=list(mnist$train$x), y=mnist$train$y); setkey(dt, y)
dt <- dt[dt[, .I[1:sizeTrain], by = y]$V1]
# dt[, .SD[1:10], by=y] # too slow
# dt %>% top_n(2, y) # even slower
x_train <- dt$x
y_train <- dt$y

batch_size <- 128
epochs <- 30




dtTest <- data.table(x=list(mnist$test$x), y=mnist$test$y); setkey(dtTest, y)
dtTest <- dtTest[, .SD[1:sizeVal], by=y, .SDcols="y"]


x_train <- dtTrain$x
y_train <- dtTrain$y

x_test <- dtTest$x
y_test <- dtTest$y



# reshape = # Redefine  dimension of train/test inputs
# x_train <- array_reshape(x_train, c(nrow(x_train), 784))
#x_test2 <- array_reshape(x_test, c(nrow(x_test), 784))
x_train <- array_reshape(x_train, c(nrow(x_train), img_rows, img_cols, 1))
x_test <- array_reshape(x_test, c(nrow(x_test), img_rows, img_cols, 1))
input_shape <- c(img_rows, img_cols, 1) # Not use in mnist_mlp.R, but use in in mnist_cnn.R (1-2-4. 1)

# rescale = # # Transform RGB values into [0,1] range
x_train <- x_train / 255;
x_test2 <- x_test2 / 255

# Binarize output (as done in PINN) = # Convert class vectors to binary class matrices
y_train <- to_categorical(y_train, num_classes)
y_test <- to_categorical(y_test, num_classes)




model <- keras_model_sequential() %>%
  layer_dense(units = 256, activation = 'relu', input_shape = c(784)) %>%
  layer_dropout(rate = 0.4) %>%
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 10, activation = 'softmax')

summary(model)


model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy')
)


history <- model %>% fit(
  x_train, y_train,
  epochs = epochs, batch_size = batch_size,
  validation_split = 0.2
)

plot(history)


score <- model %>% evaluate(
  x_test, y_test,
  verbose = 0
)

# Output metrics
cat('Test loss:', score[[1]], '\n')
cat('Test accuracy:', score[[2]], '\n')




# 1-2. Guide to the Sequential Model - https://keras.rstudio.com/articles/sequential_model.html ####


# 1-1. Training Visualization -         https://keras.rstudio.com/articles/training_visualization.html ####



# >>> 2. Other keras packages  ####


############################################################################################ #
# . library("kerasformula")  ----
# https://tensorflow.rstudio.com/blog/analyzing-rtweet-data-with-kerasformula.html
############################################################################################# #

# regression-style interface to keras_model_sequential that uses formulas and sparse matrices.

library("kerasformula")
library("rtweet")

rstats <- search_tweets("#rstats", n = 10000, include_rts = FALSE)




#  https://tensorflow.rstudio.com/blog/tensorflow-estimators-for-r.html ---

devtools::install_github("rstudio/tfestimators")

############################################################################################ #
# . library(tfruns) ----
# https://tensorflow.rstudio.com/blog/tfruns.html
############################################################################################# #

# The tfruns package provides a suite of tools for tracking, visualizing, and managing TensorFlow training runs and experiments from R.

devtools::install_github("rstudio/tfruns")


library(tfruns)
training_run("mnist_mlp.R")

latest_run()


################################################################# #



# source("mnist_mlp.R") used in tfruns.html  -----------------------------------------------
# https://github.com/rstudio/tfruns/blob/master/inst/examples/mnist_mlp/mnist_mlp.R

if (T) {

  #' Trains a simple deep NN on the MNIST dataset.
  #'
  #' Gets to 98.40% test accuracy after 20 epochs (there is *a lot* of margin for
  #' parameter tuning).
  #'

  library(keras)

  # . Hyperparameter flags ---------------------------------------------------

  FLAGS <- flags(
    flag_numeric("dropout1", 0.4),
    flag_numeric("dropout2", 0.3)
  )

  # . Data Preparation ---------------------------------------------------

  # The data, shuffled and split between train and test sets
  mnist <- dataset_mnist()
  x_train <- mnist$train$x
  y_train <- mnist$train$y
  x_test <- mnist$test$x
  y_test <- mnist$test$y

  # Reshape
  dim(x_train) <- c(nrow(x_train), 784)
  dim(x_test) <- c(nrow(x_test), 784)

  # Transform RGB values into [0,1] range
  x_train <- x_train / 255
  x_test <- x_test / 255

  # Convert class vectors to binary class matrices
  y_train <- to_categorical(y_train, 10)
  y_test <- to_categorical(y_test, 10)

  # . Define Model --------------------------------------------------------------

  model <- keras_model_sequential()
  model %>%
    layer_dense(units = 256, activation = 'relu', input_shape = c(784)) %>%
    layer_dropout(rate = FLAGS$dropout1) %>%
    layer_dense(units = 128, activation = 'relu') %>%
    layer_dropout(rate = FLAGS$dropout2) %>%
    layer_dense(units = 10, activation = 'softmax')

  model %>% compile(
    loss = 'categorical_crossentropy',
    optimizer = optimizer_rmsprop(lr = 0.001),
    metrics = c('accuracy')
  )

  # . Training & Evaluation ----------------------------------------------------

  history <- model %>% fit(
    x_train, y_train,
    batch_size = 128,
    epochs = 20,
    verbose = 1,
    validation_split = 0.2
  )

  plot(history)

  score <- model %>% evaluate(
    x_test, y_test,
    verbose = 0
  )

  cat('Test loss:', score$loss, '\n')
  cat('Test accuracy:', score$acc, '\n')



}



############################################################# #


# >>>> from https://tensorflow.rstudio.com ----
#
# # Function references:
# # https://tensorflow.rstudio.com/keras/reference/compile.html
# # https://keras.rstudio.com/
#
# Python tutorials:
#
#
# 1b=6) Python for RNN - See also: see www.tensorflow.org/tutorials/recurrent.
# https://livebook.manning.com/#!/book/machine-learning-with-tensorflow/chapter-10/35
# To understand how to implement LSTM from scratch, https://apaszke.github.io/lstm-explained.html.
#
# RNN to build A predictive model for time-series data :
# international airline passengers dataset: http://mng.bz/5UWL.
#
# [[ 6b) NO https://livebook.manning.com/#!/book/the-quick-python-book-third-edition/chapter-4
# [[ 6c) https://livebook.manning.com/#!/book/the-quick-python-book-third-edition/chapter-8/v-8/1
#
#
#
# 3) https://www.datacamp.com/community/tutorials/keras-r-deep-learning
# Jun 19, 2017
# https://towardsdatascience.com/how-to-implement-deep-learning-in-r-using-keras-and-tensorflow-82d135ae4889
# https://github.com/anishsingh20.
# This article will talk about implementing Deep learning in R on cifar10 data-set and train a Convolution Neural Network(CNN) model to classify 10,000 test images
# across 10 classes in R using Keras and Tensorflow packages.
#
#
#
#
#



# Arrays in R:

# 10 5x2 letters represented as 3D array (aka tensor)
y=1:10 # labels
x <- 1:(10*5*2); dim(x) <- c(10,2,5) # letters
x[1,,]
array_reshape(x,c(10,2*5))
array_reshape(x,c(10,5,2))
#compare to
x

# tensor -> data.frame
dt0 <- data.table(x=lapply(seq(dim(x)[1]), function(xxx) x[xxx, ,]), y=y)
dt0 <- data.table( x=tensor3D.as.list2D (x), y=y)

tensor2dt <- function(x_tensor,y) {
  data.table(x=lapply(seq(dim(x_tensor)[1]), function(xxx) x_tensor[xxx, ,]), y=y)
}

tensor3D.as.list2D <- function(x_tensor) {
  lapply(seq(dim(x_tensor)[1]), function(xxx) x_tensor[xxx, ,])
}

dt <- data.table( x=tensor3D.as.list2D (mnist$train$x), y= mnist$train$y)


dt <- tensor2dt(mnist$test$x,

# data.frame -> tensor
