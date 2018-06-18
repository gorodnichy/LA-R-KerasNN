# RStudio-keras-00-ToC+MNIST.R


# `Learn and Apply Keras/Tensorflow in R/RStudio Efficiently!`  <<< ####
# Using complete prewritten codes and your own data, With focus on analysis of sequential and text data
#
# Source: github.com/gorodnichy/LA-R-KerasNN
#
# Based on:
# - https://github.com/jjallaire/deep-learning-with-r-notebooks
# - https://keras.rstudio.com (= https://tensorflow.rstudio.com/keras)
# - https://www.manning.com/books/deep-learning-with-r
# - everything else found useful on the Web
# - and my own work back from 1995-2005 :)
#
# My Vision for AI Scientist in XXI c:
# 1. In XXI c., we (AI Scientists) will learn and program differently as we did in XX c.
# - We are the users of the IDE that we are developing ourselves and for ourselves
# - R/RStudio examplies and leads this vision
# 2. There's increasingly too much information to learn and apply -
#     "It is not what you've already learnt  but how much more can you learn what matters"
# 3. Automation everywhere, including writting reports and leveraging other's codes
# 4. S.O.S: Simplify, Organize, Subscribe  - to make everything searchable, modular, scalable and reusable
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
# See also `Learn and Apply interactive programming and presentations with RStudio Shiny`` (github.com/gorodnichy/LA-R-Rmd-Shiny)

# Contents: ####
#
# 1. Start here: https://keras.rstudio.com/index.html (which is the same as  https://tensorflow.rstudio.com/keras)
# Then, as instructed there go to. # Learning More:

# 1-2. Guide to the Sequential Model - https://keras.rstudio.com/articles/sequential_model.html
# Then, as instructed there go to # Examples:

# [1-2-1]. CIFAR10 small images classification -      https://keras.rstudio.com/articles/examples/cifar10_cnn.html
# 1-2-2. IMDB movie review sentiment classification - https://keras.rstudio.com/articles/examples/imdb_cnn_lstm.html
# 1-2-3. Reuters newswires topic classification -     https://keras.rstudio.com/articles/examples/reuters_mlp.html
# 1-2-4. MNIST handwritten digits classification -    https://keras.rstudio.com/articles/examples/mnist_mlp.html


# 1-3. Guide to the Functional API -    https://keras.rstudio.com/articles/functional_api.html
# [1-4]. Frequently Asked Questions -   https://keras.rstudio.com/articles/faq.html
# 1-1. Training Visualization -         https://keras.rstudio.com/articles/training_visualization.html

# Other files:
# - DLwR-s6.1-RNN-for-text.R
# - DLwR-s6.1-RNN-forSequences.R
# - DLwR-s3-IMDB_sentimentBinary+wiresClassification+housepriceReression.R

library(ggplot2);library(data.table); library(magrittr);
library(tibble); library(readr); library(keras)

################################################################################################ #
################################################################################################ #

# 1. "Hello World" for Keras: MNIST 28x28 digit recognition ####
# https://keras.rstudio.com/index.html
# = https://tensorflow.rstudio.com/keras/


mnist <- dataset_mnist()
x_train <- mnist$train$x
y_train <- mnist$train$y
x_test <- mnist$test$x
y_test <- mnist$test$y


# reshape
x_train <- array_reshape(x_train, c(nrow(x_train), 784))
x_test <- array_reshape(x_test, c(nrow(x_test), 784))
# rescale
x_train <- x_train / 255
x_test <- x_test / 255


y_train <- to_categorical(y_train, 10)
y_test <- to_categorical(y_test, 10)

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
  epochs = 30, batch_size = 128,
  validation_split = 0.2
)

plot(history)


#
#
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
