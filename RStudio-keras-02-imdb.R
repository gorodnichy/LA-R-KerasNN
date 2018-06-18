# RStudio-keras-01-mnist.R
#
# From:
# https://github.com/rstudio/keras/blob/master/vignettes/examples/imdb_cnn.R =
# https://keras.rstudio.com/articles/examples/imdb_cnn.html =
# https://tensorflow.rstudio.com/keras/articles/examples/imdb_cnn.html
#



#.################################################################################
# 1 > imdb_cnn.R ####
############################################################################### #


#' Use Convolution1D for text classification.
#'
#' Output after 2 epochs: ~0.89
#' Time per epoch on CPU (Intel i5 2.4Ghz): 90s
#' Time per epoch on GPU (Tesla K40): 10s
#'

library(keras)

# Set parameters:
max_features <- 5000
maxlen <- 400
batch_size <- 32
embedding_dims <- 50
filters <- 250
kernel_size <- 3
hidden_dims <- 250
epochs <- 2


# Data Preparation --------------------------------------------------------

# Keras load all data into a list with the following structure:
# List of 2
# $ train:List of 2
# ..$ x:List of 25000
# .. .. [list output truncated]
# .. ..- attr(*, "dim")= int 25000
# ..$ y: num [1:25000(1d)] 1 0 0 1 0 0 1 0 1 0 ...
# $ test :List of 2
# ..$ x:List of 25000
# .. .. [list output truncated]
# .. ..- attr(*, "dim")= int 25000
# ..$ y: num [1:25000(1d)] 1 1 1 1 1 0 0 0 1 1 ...
#
# The x data includes integer sequences, each integer is a word.
# The y data includes a set of integer labels (0 or 1).
# The num_words argument indicates that only the max_fetures most frequent
# words will be integerized. All other will be ignored.
# See help(dataset_imdb)
imdb <- dataset_imdb(num_words = max_features)

# Pad the sequences, so they have all the same length
# This will convert the dataset into a matrix: each line is a review
# and each column a word on the sequence.
# Pad the sequences with 0 to the left.
x_train <- imdb$train$x %>%
  pad_sequences(maxlen = maxlen)
x_test <- imdb$test$x %>%
  pad_sequences(maxlen = maxlen)

# Defining Model ------------------------------------------------------

#Initialize model
model <- keras_model_sequential()

model %>%
  # Start off with an efficient embedding layer which maps
  # the vocab indices into embedding_dims dimensions
  layer_embedding(max_features, embedding_dims, input_length = maxlen) %>%
  layer_dropout(0.2) %>%

  # Add a Convolution1D, which will learn filters
  # Word group filters of size filter_length:
  layer_conv_1d(
    filters, kernel_size,
    padding = "valid", activation = "relu", strides = 1
  ) %>%
  # Apply max pooling:
  layer_global_max_pooling_1d() %>%

  # Add a vanilla hidden layer:
  layer_dense(hidden_dims) %>%

  # Apply 20% layer dropout
  layer_dropout(0.2) %>%
  layer_activation("relu") %>%

  # Project onto a single unit output layer, and squash it with a sigmoid

  layer_dense(1) %>%
  layer_activation("sigmoid")

# Compile model
model %>% compile(
  loss = "binary_crossentropy",
  optimizer = "adam",
  metrics = "accuracy"
)

# Training ----------------------------------------------------------------

model %>%
  fit(
    x_train, imdb$train$y,
    batch_size = batch_size,
    epochs = epochs,
    validation_data = list(x_test, imdb$test$y)
  )





# . ################################################################################
# 2 > imdb_lstm.R ####
############################################################################### #

#' Trains a LSTM on the IMDB sentiment classification task.
#'
#' The dataset is actually too small for LSTM to be of any advantage compared to
#' simpler, much faster methods such as TF-IDF + LogReg.
#'
#' Notes:
#' - RNNs are tricky. Choice of batch size is important, choice of loss and
#'   optimizer is critical, etc. Some configurations won't converge.
#' - LSTM loss decrease patterns during training can be quite different from
#'   what you see with CNNs/MLPs/etc.

library(keras)

max_features <- 20000
batch_size <- 32

# Cut texts after this number of words (among top max_features most common words)
maxlen <- 80

cat('Loading data...\n')
imdb <- dataset_imdb(num_words = max_features)
x_train <- imdb$train$x
y_train <- imdb$train$y
x_test <- imdb$test$x
y_test <- imdb$test$y

cat(length(x_train), 'train sequences\n')
cat(length(x_test), 'test sequences\n')

cat('Pad sequences (samples x time)\n')
x_train <- pad_sequences(x_train, maxlen = maxlen)
x_test <- pad_sequences(x_test, maxlen = maxlen)
cat('x_train shape:', dim(x_train), '\n')
cat('x_test shape:', dim(x_test), '\n')

cat('Build model...\n')
model <- keras_model_sequential()
model %>%
  layer_embedding(input_dim = max_features, output_dim = 128) %>%
  layer_lstm(units = 64, dropout = 0.2, recurrent_dropout = 0.2) %>%
  layer_dense(units = 1, activation = 'sigmoid')

# Try using different optimizers and different optimizer configs
model %>% compile(
  loss = 'binary_crossentropy',
  optimizer = 'adam',
  metrics = c('accuracy')
)

cat('Train...\n')
model %>% fit(
  x_train, y_train,
  batch_size = batch_size,
  epochs = 15,
  validation_data = list(x_test, y_test)
)

scores <- model %>% evaluate(
  x_test, y_test,
  batch_size = batch_size
)

cat('Test score:', scores[[1]])
cat('Test accuracy', scores[[2]])





# . ################################################################################
# 3 > imdb_cnn_lstm.R ####
############################################################################### #


#' Train a recurrent convolutional network on the IMDB sentiment
#' classification task.
#'
#' Achieves 0.8498 test accuracy after 2 epochs. 41s/epoch on K520 GPU.

library(keras)

# Parameters --------------------------------------------------------------

# Embedding
max_features = 20000
maxlen = 100
embedding_size = 128

# Convolution
kernel_size = 5
filters = 64
pool_size = 4

# LSTM
lstm_output_size = 70

# Training
batch_size = 30
epochs = 2

# Data Preparation --------------------------------------------------------

# The x data includes integer sequences, each integer is a word
# The y data includes a set of integer labels (0 or 1)
# The num_words argument indicates that only the max_fetures most frequent
# words will be integerized. All other will be ignored.
# See help(dataset_imdb)
imdb <- dataset_imdb(num_words = max_features)
# Keras load all data into a list with the following structure:
str(imdb)

# Pad the sequences to the same length
# This will convert our dataset into a matrix: each line is a review
# and each column a word on the sequence
# We pad the sequences with 0s to the left
x_train <- imdb$train$x %>%
  pad_sequences(maxlen = maxlen)
x_test <- imdb$test$x %>%
  pad_sequences(maxlen = maxlen)

# Defining Model ------------------------------------------------------

model <- keras_model_sequential()

model %>%
  layer_embedding(max_features, embedding_size, input_length = maxlen) %>%
  layer_dropout(0.25) %>%
  layer_conv_1d(
    filters,
    kernel_size,
    padding = "valid",
    activation = "relu",
    strides = 1
  ) %>%
  layer_max_pooling_1d(pool_size) %>%
  layer_lstm(lstm_output_size) %>%
  layer_dense(1) %>%
  layer_activation("sigmoid")

model %>% compile(
  loss = "binary_crossentropy",
  optimizer = "adam",
  metrics = "accuracy"
)

# Training ----------------------------------------------------------------

model %>% fit(
  x_train, imdb$train$y,
  batch_size = batch_size,
  epochs = epochs,
  validation_data = list(x_test, imdb$test$y)
)







# . ################################################################################
# 4 > imdb_bidirectional_lstm.R ####
############################################################################### #


#' Train a Bidirectional LSTM on the IMDB sentiment classification task.
#'
#' Output after 4 epochs on CPU: ~0.8146
#' Time per epoch on CPU (Core i7): ~150s.

library(keras)

# Define maximum number of input features
max_features <- 20000

# Cut texts after this number of words
# (among top max_features most common words)
maxlen <- 100

batch_size <- 32

# Load imdb dataset
cat('Loading data...\n')
imdb <- dataset_imdb(num_words = max_features)

# Define training and test sets
x_train <- imdb$train$x
y_train <- imdb$train$y
x_test <- imdb$test$x
y_test <- imdb$test$y

# Output lengths of testing and training sets
cat(length(x_train), 'train sequences\n')
cat(length(x_test), 'test sequences\n')

cat('Pad sequences (samples x time)\n')

# Pad training and test inputs
x_train <- pad_sequences(x_train, maxlen = maxlen)
x_test <- pad_sequences(x_test, maxlen = maxlen)

# Output dimensions of training and test inputs
cat('x_train shape:', dim(x_train), '\n')
cat('x_test shape:', dim(x_test), '\n')

# Initialize model
model <- keras_model_sequential()
model %>%
  # Creates dense embedding layer; outputs 3D tensor
  # with shape (batch_size, sequence_length, output_dim)
  layer_embedding(input_dim = max_features,
                  output_dim = 128,
                  input_length = maxlen) %>%
  bidirectional(layer_lstm(units = 64)) %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 1, activation = 'sigmoid')

# Try using different optimizers and different optimizer configs
model %>% compile(
  loss = 'binary_crossentropy',
  optimizer = 'adam',
  metrics = c('accuracy')
)

# Train model over four epochs
cat('Train...\n')
model %>% fit(
  x_train, y_train,
  batch_size = batch_size,
  epochs = 4,
  validation_data = list(x_test, y_test)
)







# . ################################################################################
# 5 > imdb_fasttext.R ####
############################################################################### #


#' This example demonstrates the use of fasttext for text classification
#'
#' Based on Joulin et al's paper:
#' "Bags of Tricks for Efficient Text Classification"
#' https://arxiv.org/abs/1607.01759
#'
#' Results on IMDB datasets with uni and bi-gram embeddings:
#'  Uni-gram: 0.8813 test accuracy after 5 epochs. 8s/epoch on i7 CPU
#'  Bi-gram : 0.9056 test accuracy after 5 epochs. 2s/epoch on GTx 980M GPU
#'

library(keras)
library(purrr)

# Function Definitions ----------------------------------------------------

create_ngram_set <- function(input_list, ngram_value = 2){
  indices <- map(0:(length(input_list) - ngram_value), ~1:ngram_value + .x)
  indices %>%
    map_chr(~input_list[.x] %>% paste(collapse = "|")) %>%
    unique()
}

add_ngram <- function(sequences, token_indice, ngram_range = 2){
  ngrams <- map(
    sequences,
    create_ngram_set, ngram_value = ngram_range
  )

  seqs <- map2(sequences, ngrams, function(x, y){
    tokens <- token_indice$token[token_indice$ngrams %in% y]
    c(x, tokens)
  })

  seqs
}


# Parameters --------------------------------------------------------------

# ngram_range = 2 will add bi-grams features
ngram_range <- 2
max_features <- 20000
maxlen <- 400
batch_size <- 32
embedding_dims <- 50
epochs <- 5


# Data Preparation --------------------------------------------------------

# Load data
imdb_data <- dataset_imdb(num_words = max_features)

# Train sequences
print(length(imdb_data$train$x))
print(sprintf("Average train sequence length: %f", mean(map_int(imdb_data$train$x, length))))

# Test sequences
print(length(imdb_data$test$x))
print(sprintf("Average test sequence length: %f", mean(map_int(imdb_data$test$x, length))))

if(ngram_range > 1) {

  # Create set of unique n-gram from the training set.
  ngrams <- imdb_data$train$x %>%
    map(create_ngram_set) %>%
    unlist() %>%
    unique()

  # Dictionary mapping n-gram token to a unique integer
  # Integer values are greater than max_features in order
  # to avoid collision with existing features
  token_indice <- data.frame(
    ngrams = ngrams,
    token  = 1:length(ngrams) + (max_features),
    stringsAsFactors = FALSE
  )

  # max_features is the highest integer that could be found in the dataset
  max_features <- max(token_indice$token) + 1

  # Augmenting x_train and x_test with n-grams features
  imdb_data$train$x <- add_ngram(imdb_data$train$x, token_indice, ngram_range)
  imdb_data$test$x <- add_ngram(imdb_data$test$x, token_indice, ngram_range)
}

# Pad sequences
imdb_data$train$x <- pad_sequences(imdb_data$train$x, maxlen = maxlen)
imdb_data$test$x <- pad_sequences(imdb_data$test$x, maxlen = maxlen)


# Model Definition --------------------------------------------------------

model <- keras_model_sequential()

model %>%
  layer_embedding(
    input_dim = max_features, output_dim = embedding_dims,
    input_length = maxlen
  ) %>%
  layer_global_average_pooling_1d() %>%
  layer_dense(1, activation = "sigmoid")

model %>% compile(
  loss = "binary_crossentropy",
  optimizer = "adam",
  metrics = "accuracy"
)


# Fitting -----------------------------------------------------------------

model %>% fit(
  imdb_data$train$x, imdb_data$train$y,
  batch_size = batch_size,
  epochs = epochs,
  validation_data = list(imdb_data$test$x, imdb_data$test$y)
)


# END. ####
