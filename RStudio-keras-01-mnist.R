# RStudio-keras-01-mnist.R
#
# From:
# https://github.com/rstudio/keras/blob/master/vignettes/examples/mnist_cnn.R =
# https://keras.rstudio.com/articles/examples/mnist_cnn.html =
# https://tensorflow.rstudio.com/keras/articles/examples/mnist_cnn.html
#



#.############################################################################
# 2 > mnist_cnn.R ####


#' Trains a simple convnet on the MNIST dataset.
#'
#' Gets to 99.25% test accuracy after 12 epochs
#'  Note: There is still a large margin for parameter tuning
#'
#' 16 seconds per epoch on a GRID K520 GPU.

library(keras)

# Data Preparation -----------------------------------------------------

batch_size <- 128
num_classes <- 10
epochs <- 12

# Input image dimensions
img_rows <- 28
img_cols <- 28

# The data, shuffled and split between train and test sets
mnist <- dataset_mnist()
x_train <- mnist$train$x
y_train <- mnist$train$y
x_test <- mnist$test$x
y_test <- mnist$test$y

# Redefine  dimension of train/test inputs
x_train <- array_reshape(x_train, c(nrow(x_train), img_rows, img_cols, 1))
x_test <- array_reshape(x_test, c(nrow(x_test), img_rows, img_cols, 1))
input_shape <- c(img_rows, img_cols, 1)

# Transform RGB values into [0,1] range
x_train <- x_train / 255
x_test <- x_test / 255

cat('x_train_shape:', dim(x_train), '\n')
cat(nrow(x_train), 'train samples\n')
cat(nrow(x_test), 'test samples\n')

# Convert class vectors to binary class matrices
y_train <- to_categorical(y_train, num_classes)
y_test <- to_categorical(y_test, num_classes)

# Define Model -----------------------------------------------------------

# Define model
model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3,3), activation = 'relu',
                input_shape = input_shape) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = 'relu') %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_dropout(rate = 0.25) %>%
  layer_flatten() %>%
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = num_classes, activation = 'softmax')

# Compile model
model %>% compile(
  loss = loss_categorical_crossentropy,
  optimizer = optimizer_adadelta(),
  metrics = c('accuracy')
)

# Train model
model %>% fit(
  x_train, y_train,
  batch_size = batch_size,
  epochs = epochs,
  validation_split = 0.2
)




scores <- model %>% evaluate(
  x_test, y_test, verbose = 0
)

# Output metrics
cat('Test loss:', scores[[1]], '\n')
cat('Test accuracy:', scores[[2]], '\n')


#.####################################################################################
# 1 > mnist_mlp.html ####

# Trains a simple deep NN on the MNIST dataset.
# Gets to 98.40% test accuracy after 20 epochs (there is a lot of margin for parameter tuning). 2 seconds per epoch on a K520 GPU.

library(keras)

# Data Preparation ---------------------------------------------------

batch_size <- 128
num_classes <- 10
epochs <- 30

# The data, shuffled and split between train and test sets
c(c(x_train, y_train), c(x_test, y_test)) %<-% dataset_mnist()

x_train <- array_reshape(x_train, c(nrow(x_train), 784))
x_test <- array_reshape(x_test, c(nrow(x_test), 784))

# Transform RGB values into [0,1] range
x_train <- x_train / 255
x_test <- x_test / 255

cat(nrow(x_train), 'train samples\n')
cat(nrow(x_test), 'test samples\n')

# Convert class vectors to binary class matrices
y_train <- to_categorical(y_train, num_classes)
y_test <- to_categorical(y_test, num_classes)

# Define Model --------------------------------------------------------------

model <- keras_model_sequential()
model %>%
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

# Training & Evaluation ----------------------------------------------------

# Fit model to data
history <- model %>% fit(
  x_train, y_train,
  batch_size = batch_size,
  epochs = epochs,
  verbose = 1,
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



#.################################################################################
# 4 > mnist_antirectifier.R ####
############################################################################### #

# Demonstrates how to write custom layers for Keras.
# We build a custom activation layer called ‘Antirectifier’, which modifies the
# shape of the tensor that passes through it. We need to specify two methods: compute_output_shape and call.
# Note that the same result can also be achieved via a Lambda layer.

library(keras)

# Data Preparation --------------------------------------------------------

batch_size <- 128
num_classes <- 10
epochs <- 40

# The data, shuffled and split between train and test sets
mnist <- dataset_mnist()
x_train <- mnist$train$x
y_train <- mnist$train$y
x_test <- mnist$test$x
y_test <- mnist$test$y

# Redimension
x_train <- array_reshape(x_train, c(nrow(x_train), 784))
x_test <- array_reshape(x_test, c(nrow(x_test), 784))

# Transform RGB values into [0,1] range
x_train <- x_train / 255
x_test <- x_test / 255

cat(nrow(x_train), 'train samples\n')
cat(nrow(x_test), 'test samples\n')

# Convert class vectors to binary class matrices
y_train <- to_categorical(y_train, num_classes)
y_test <- to_categorical(y_test, num_classes)

# Antirectifier Layer -----------------------------------------------------
This is the combination of a sample-wise L2 normalization with the concatenation of the positive part of the input with the negative part of the input. The result is a tensor of samples that are twice as large as the input samples.

It can be used in place of a ReLU. Input shape: 2D tensor of shape (samples, n) Output shape: 2D tensor of shape (samples, 2*n)

When applying ReLU, assuming that the distribution of the previous output is approximately centered around 0., you are discarding half of your input. This is inefficient.

Antirectifier allows to return all-positive outputs like ReLU, without discarding any data.

Tests on MNIST show that Antirectifier allows to train networks with half the parameters yet with comparable classification accuracy as an equivalent ReLU-based network.

# Custom layer class
AntirectifierLayer <- R6::R6Class("KerasLayer",

                                  inherit = KerasLayer,

                                  public = list(

                                    call = function(x, mask = NULL) {
                                      x <- x - k_mean(x, axis = 2, keepdims = TRUE)
                                      x <- k_l2_normalize(x, axis = 2)
                                      pos <- k_relu(x)
                                      neg <- k_relu(-x)
                                      k_concatenate(c(pos, neg), axis = 2)

                                    },

                                    compute_output_shape = function(input_shape) {
                                      input_shape[[2]] <- input_shape[[2]] * 2L
                                      tuple(input_shape)
                                    }
                                  )
)

# Create layer wrapper function
layer_antirectifier <- function(object) {
  create_layer(AntirectifierLayer, object)
}


# Define & Train Model -------------------------------------------------

model <- keras_model_sequential()
model %>%
  layer_dense(units = 256, input_shape = c(784)) %>%
  layer_antirectifier() %>%
  layer_dropout(rate = 0.1) %>%
  layer_dense(units = 256) %>%
  layer_antirectifier() %>%
  layer_dropout(rate = 0.1) %>%
  layer_dense(units = num_classes, activation = 'softmax')

# Compile the model
model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = 'rmsprop',
  metrics = c('accuracy')
)

# Train the model
model %>% fit(x_train, y_train,
              batch_size = batch_size,
              epochs = epochs,
              verbose = 1,
              validation_data= list(x_test, y_test)
)


#.################################################################################
# 3 > mnist_irnn ####
############################################################################### #

#' This is a reproduction of the IRNN experiment with pixel-by-pixel sequential
#' MNIST in "A Simple Way to Initialize Recurrent Networks of Rectified Linear Units"
#' by Quoc V. Le, Navdeep Jaitly, Geoffrey E. Hinton
#'
#' arxiv:1504.00941v2 [cs.NE] 7 Apr 2015
#' http://arxiv.org/pdf/1504.00941v2.pdf
#'
#' Optimizer is replaced with RMSprop which yields more stable and steady
#' improvement.
#'
#' Reaches 0.93 train/test accuracy after 900 epochs
#' This corresponds to roughly 1687500 steps in the original paper.

library(keras)

# Data Preparation ---------------------------------------------------------------

batch_size <- 32
num_classes <- 10
epochs <- 200
hidden_units <- 100

img_rows <- 28
img_cols <- 28

learning_rate <- 1e-6
clip_norm <- 1.0

# The data, shuffled and split between train and test sets
mnist <- dataset_mnist()
x_train <- mnist$train$x
y_train <- mnist$train$y
x_test <- mnist$test$x
y_test <- mnist$test$y

x_train <- array_reshape(x_train, c(nrow(x_train), img_rows * img_cols, 1))
x_test <- array_reshape(x_test, c(nrow(x_test), img_rows * img_cols, 1))
input_shape <- c(img_rows, img_cols, 1)

# Transform RGB values into [0,1] range
x_train <- x_train / 255
x_test <- x_test / 255

cat('x_train_shape:', dim(x_train), '\n')
cat(nrow(x_train), 'train samples\n')
cat(nrow(x_test), 'test samples\n')

# Convert class vectors to binary class matrices
y_train <- to_categorical(y_train, num_classes)
y_test <- to_categorical(y_test, num_classes)

# Define Model ------------------------------------------------------------------

model <- keras_model_sequential()
model %>%
  layer_simple_rnn(units = hidden_units,
                   kernel_initializer = initializer_random_normal(stddev = 0.01),
                   recurrent_initializer = initializer_identity(gain = 1.0),
                   activation = 'relu',
                   input_shape = dim(x_train)[-1]) %>%
  layer_dense(units = num_classes) %>%
  layer_activation(activation = 'softmax')

model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(lr = learning_rate),
  metrics = c('accuracy')
)

# Training & Evaluation ---------------------------------------------------------

cat("Evaluate IRNN...\n")
model %>% fit(
  x_train, y_train,
  batch_size = batch_size,
  epochs = epochs,
  verbose = 1,
  validation_data = list(x_test, y_test)
)

scores <- model %>% evaluate(x_test, y_test, verbose = 0)
cat('IRNN test score:', scores[[1]], '\n')
cat('IRNN test accuracy:', scores[[2]], '\n')



#.################################################################################
# 5 > mnist_hierarchical_rnn.R ####
############################################################################### #

#' This is an example of using Hierarchical RNN (HRNN) to classify MNIST digits.
#'
#' HRNNs can learn across multiple levels of temporal hiearchy over a complex sequence.
#' Usually, the first recurrent layer of an HRNN encodes a sentence (e.g. of word vectors)
#' into a  sentence vector. The second recurrent layer then encodes a sequence of
#' such vectors (encoded by the first layer) into a document vector. This
#' document vector is considered to preserve both the word-level and
#' sentence-level structure of the context.
#'
#' References:
#' - [A Hierarchical Neural Autoencoder for Paragraphs and Documents](https://arxiv.org/abs/1506.01057)
#'   Encodes paragraphs and documents with HRNN.
#'   Results have shown that HRNN outperforms standard RNNs and may play some role in more
#'   sophisticated generation tasks like summarization or question answering.
#' - [Hierarchical recurrent neural network for skeleton based action recognition](http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7298714)
#'   Achieved state-of-the-art results on skeleton based action recognition with 3 levels
#'   of bidirectional HRNN combined with fully connected layers.
#'
#' In the below MNIST example the first LSTM layer first encodes every
#' column of pixels of shape (28, 1) to a column vector of shape (128,). The second LSTM
#' layer encodes then these 28 column vectors of shape (28, 128) to a image vector
#' representing the whole image. A final dense layer is added for prediction.
#'
#' After 5 epochs: train acc: 0.9858, val acc: 0.9864
#'

library(keras)

# Data Preparation -----------------------------------------------------------------

# Training parameters.
batch_size <- 32
num_classes <- 10
epochs <- 5

# Embedding dimensions.
row_hidden <- 128
col_hidden <- 128

# The data, shuffled and split between train and test sets
mnist <- dataset_mnist()
x_train <- mnist$train$x
y_train <- mnist$train$y
x_test <- mnist$test$x
y_test <- mnist$test$y

# Reshapes data to 4D for Hierarchical RNN.
x_train <- array_reshape(x_train, c(nrow(x_train), 28, 28, 1))
x_test <- array_reshape(x_test, c(nrow(x_test), 28, 28, 1))
x_train <- x_train / 255
x_test <- x_test / 255

dim_x_train <- dim(x_train)
cat('x_train_shape:', dim_x_train)
cat(nrow(x_train), 'train samples')
cat(nrow(x_test), 'test samples')

# Converts class vectors to binary class matrices
y_train <- to_categorical(y_train, num_classes)
y_test <- to_categorical(y_test, num_classes)

# Define input dimensions
row <- dim_x_train[[2]]
col <- dim_x_train[[3]]
pixel <- dim_x_train[[4]]

# Model input (4D)
input <- layer_input(shape = c(row, col, pixel))

# Encodes a row of pixels using TimeDistributed Wrapper
encoded_rows <- input %>% time_distributed(layer_lstm(units = row_hidden))

# Encodes columns of encoded rows
encoded_columns <- encoded_rows %>% layer_lstm(units = col_hidden)

# Model output
prediction <- encoded_columns %>%
  layer_dense(units = num_classes, activation = 'softmax')

# Define Model ------------------------------------------------------------------------

model <- keras_model(input, prediction)
model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = 'rmsprop',
  metrics = c('accuracy')
)

# Training
model %>% fit(
  x_train, y_train,
  batch_size = batch_size,
  epochs = epochs,
  verbose = 1,
  validation_data = list(x_test, y_test)
)

# Evaluation
scores <- model %>% evaluate(x_test, y_test, verbose = 0)
cat('Test loss:', scores[[1]], '\n')
cat('Test accuracy:', scores[[2]], '\n')



#.################################################################################
# 6 > mnist_transfer_cnn.R ####
############################################################################### #


#' Transfer learning toy example:
#'
#' 1) Train a simple convnet on the MNIST dataset the first 5 digits [0..4].
#' 2) Freeze convolutional layers and fine-tune dense layers
#'    for the classification of digits [5..9].
#'

library(keras)

now <- Sys.time()

batch_size <- 128
num_classes <- 5
epochs <- 5

# input image dimensions
img_rows <- 28
img_cols <- 28

# number of convolutional filters to use
filters <- 32

# size of pooling area for max pooling
pool_size <- 2

# convolution kernel size
kernel_size <- c(3, 3)

# input shape
input_shape <- c(img_rows, img_cols, 1)

# the data, shuffled and split between train and test sets
data <- dataset_mnist()
x_train <- data$train$x
y_train <- data$train$y
x_test <- data$test$x
y_test <- data$test$y

# create two datasets one with digits below 5 and one with 5 and above
x_train_lt5 <- x_train[y_train < 5]
y_train_lt5 <- y_train[y_train < 5]
x_test_lt5 <- x_test[y_test < 5]
y_test_lt5 <- y_test[y_test < 5]

x_train_gte5 <- x_train[y_train >= 5]
y_train_gte5 <- y_train[y_train >= 5] - 5
x_test_gte5 <- x_test[y_test >= 5]
y_test_gte5 <- y_test[y_test >= 5] - 5

# define two groups of layers: feature (convolutions) and classification (dense)
feature_layers <-
  layer_conv_2d(filters = filters, kernel_size = kernel_size,
                input_shape = input_shape) %>%
  layer_activation(activation = 'relu') %>%
  layer_conv_2d(filters = filters, kernel_size = kernel_size) %>%
  layer_activation(activation = 'relu') %>%
  layer_max_pooling_2d(pool_size = pool_size) %>%
  layer_dropout(rate = 0.25) %>%
  layer_flatten()



# feature_layers = [
#   Conv2D(filters, kernel_size,
#          padding='valid',
#          input_shape=input_shape),
#   Activation('relu'),
#   Conv2D(filters, kernel_size),
#   Activation('relu'),
#   MaxPooling2D(pool_size=pool_size),
#   Dropout(0.25),
#   Flatten(),
#   ]
#
# classification_layers = [
#   Dense(128),
#   Activation('relu'),
#   Dropout(0.5),
#   Dense(num_classes),
#   Activation('softmax')
#   ]


#.################################################################################
# 0 > cifar10_cnn.R ####
############################################################################### #

#' Train a simple deep CNN on the CIFAR10 small images dataset.
#'
#' It gets down to 0.65 test logloss in 25 epochs, and down to 0.55 after 50 epochs,
#' though it is still underfitting at that point.

library(keras)

# Parameters --------------------------------------------------------------

batch_size <- 32
epochs <- 200
data_augmentation <- TRUE


# Data Preparation --------------------------------------------------------

# See ?dataset_cifar10 for more info
cifar10 <- dataset_cifar10()

# Feature scale RGB values in test and train inputs
x_train <- cifar10$train$x/255
x_test <- cifar10$test$x/255
y_train <- to_categorical(cifar10$train$y, num_classes = 10)
y_test <- to_categorical(cifar10$test$y, num_classes = 10)


# Defining Model ----------------------------------------------------------

# Initialize sequential model
model <- keras_model_sequential()

model %>%

  # Start with hidden 2D convolutional layer being fed 32x32 pixel images
  layer_conv_2d(
    filter = 32, kernel_size = c(3,3), padding = "same",
    input_shape = c(32, 32, 3)
  ) %>%
  layer_activation("relu") %>%

  # Second hidden layer
  layer_conv_2d(filter = 32, kernel_size = c(3,3)) %>%
  layer_activation("relu") %>%

  # Use max pooling
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_dropout(0.25) %>%

  # 2 additional hidden 2D convolutional layers
  layer_conv_2d(filter = 32, kernel_size = c(3,3), padding = "same") %>%
  layer_activation("relu") %>%
  layer_conv_2d(filter = 32, kernel_size = c(3,3)) %>%
  layer_activation("relu") %>%

  # Use max pooling once more
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_dropout(0.25) %>%

  # Flatten max filtered output into feature vector
  # and feed into dense layer
  layer_flatten() %>%
  layer_dense(512) %>%
  layer_activation("relu") %>%
  layer_dropout(0.5) %>%

  # Outputs from dense layer are projected onto 10 unit output layer
  layer_dense(10) %>%
  layer_activation("softmax")

opt <- optimizer_rmsprop(lr = 0.0001, decay = 1e-6)

model %>% compile(
  loss = "categorical_crossentropy",
  optimizer = opt,
  metrics = "accuracy"
)


# Training ----------------------------------------------------------------

if(!data_augmentation){

  model %>% fit(
    x_train, y_train,
    batch_size = batch_size,
    epochs = epochs,
    validation_data = list(x_test, y_test),
    shuffle = TRUE
  )

} else {

  datagen <- image_data_generator(
    featurewise_center = TRUE,
    featurewise_std_normalization = TRUE,
    rotation_range = 20,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    horizontal_flip = TRUE
  )

  datagen %>% fit_image_data_generator(x_train)

  model %>% fit_generator(
    flow_images_from_data(x_train, y_train, datagen, batch_size = batch_size),
    steps_per_epoch = as.integer(50000/batch_size),
    epochs = epochs,
    validation_data = list(x_test, y_test)
  )

}




# END ####




