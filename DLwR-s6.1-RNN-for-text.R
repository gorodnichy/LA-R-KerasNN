# DLwR-s6.1-RNN-for-text.R
# https://livebook.manning.com/#!/book/deep-learning-with-r/chapter-6
# >>> Chapter 6. Deep learning for text and sequences (Part 1: RNN, LSTM, [GRU] - for IMDB movie data) ####
#
# Also:
# 2.2) https://tensorflow.rstudio.com/blog/word-embeddings-with-keras.html
# 2.3) https://tensorflow.rstudio.com/blog/text-classification-with-keras.html

# >>> See also <>  https://statsmaths.github.io/stat395-f17/class23/ ####
# Class 23: Lions, Tigres, and 狗熊 (oh my)
# https://statsmaths.github.io/stat395-f17/class22/ - Class 22: Vector Representations of Words
# https://statsmaths.github.io/stat395-f17/class20/ - Class 20: Faster, Higher, Stronger (and Deeper!)
# https://statsmaths.github.io/blog/state-of-union-case-study/ - A Case Study with cleanNLP, 1 July 2017
#

# https://arxiv.org/pdf/1301.3781v3.pdf - gnomal paper Efficient Estimation of Word Representations in Vector Space


## > 6.1-one-hot-encoding-of-words-or-characters.nb.html ####



# Listing 6.1. Word-level one-hot encoding (toy example) ####

samples <- c("The cat sat on the mat.", "The dog ate my homework.")

# First, build an index of all tokens in the data.
token_index <- list()
for (sample in samples) {
  # Tokenizes the samples via the strsplit function. In real life, you'd also
  # strip punctuation and special characters from the samples.
  for (word in strsplit(sample, " ")[[1]])
    if (!word %in% names(token_index))
      # Assigns a unique index to each unique word. Note that you don't
      # attribute index 1 to anything.
      token_index[[word]] <- length(token_index) + 2
    # Vectorizes the samples. You'll only consider the first max_length
    # words in each sample.
    max_length <- 10
    # This is where you store the results.
    results <- array(0, dim = c(length(samples),
                                max_length,
                                max(as.integer(token_index))))
    for (i in 1:length(samples)) {
      sample <- samples[[i]]
      words <- head(strsplit(sample, " ")[[1]], n = max_length)
      for (j in 1:length(words)) {
        index <- token_index[[words[[j]]]]
        results[[i, j, index]] <- 1
      }
    }
}


# Listing 6.2. Character-level one-hot encoding (toy example) ####

ascii_tokens <- c("", sapply(as.raw(c(32:126)), rawToChar))
token_index <- c(1:(length(ascii_tokens)))
names(token_index) <- ascii_tokens
max_length <- 50
results <- array(0, dim = c(length(samples), max_length, length(token_index)))
for (i in 1:length(samples)) {
  sample <- samples[[i]]
  characters <- strsplit(sample, "")[[1]]
  for (j in 1:length(characters)) {
    character <- characters[[j]]
    results[i, j, token_index[[character]]] <- 1
  }
}


# Listing 6.3. Using Keras for word-level one-hot encoding ####

library(keras)

tokenizer <- text_tokenizer(num_words = 1000) %>%
  fit_text_tokenizer(samples)
# Turns strings into lists of integer indices
sequences <- texts_to_sequences(tokenizer, samples)
# You could also directly get the one-hot binary representations. Vectorization
# modes other than one-hot encoding are supported by this tokenizer.
one_hot_results <- texts_to_matrix(tokenizer, samples, mode = "binary")
# How you can recover the word index that was computed
word_index <- tokenizer$word_index
cat("Found", length(word_index), "unique tokens.\n")



## > 6.1-using-word-embeddings.nb.html ####

## Another popular and powerful way to associate a vector with a word is the use of dense “word vectors”, also called “word embeddings”.

# Listing 6.5. Instantiating an embedding layer  ####
#
# The embedding layer takes at least two arguments:
# the number of possible tokens, here 1000 (1 + maximum word index),
# and the dimensionality of the embeddings, here 64.
embedding_layer <- layer_embedding(input_dim = 1000, output_dim = 64)

# Listing 6.6. Loading the IMDB data for use with an embedding layer #####

# Number of words to consider as features
max_features <- 1000
# Cut texts after this number of words
# (among top max_features most common words)
maxlen <- 20
# Load the data as lists of integers.
imdb <- dataset_imdb(num_words = max_features)
c(c(x_train, y_train), c(x_test, y_test)) %<-% imdb
# This turns our lists of integers
# into a 2D integer tensor of shape `(samples, maxlen)`
x_train <- pad_sequences(x_train, maxlen = maxlen)
x_test <- pad_sequences(x_test, maxlen = maxlen)


# Listing 6.7. Using an embedding layer and classifier on the IMDB data ####

model <- keras_model_sequential() %>%
  # We specify the maximum input length to our Embedding layer
  # so we can later flatten the embedded inputs
  layer_embedding(input_dim = 1000, output_dim = 8,
                  input_length = maxlen) %>%
  # We flatten the 3D tensor of embeddings
  # into a 2D tensor of shape `(samples, maxlen * 8)`
  layer_flatten() %>%
  # We add the classifier on top
  layer_dense(units = 1, activation = "sigmoid")
model %>% compile(
  optimizer = "rmsprop",
  loss = "binary_crossentropy",
  metrics = c("acc")
)
history <- model %>% fit(
  x_train, y_train,
  epochs = 10,
  batch_size = 32,
  validation_split = 0.2
)



# Listing 6.8. Processing the labels of the raw IMDB data ####

# Yoshua Bengio et al., Neural Probabilistic Language Models (Springer, 2003).
# https://nlp.stanford.edu/projects/glove),
# https://code.google.com/archive/p/word2vec)

imdb_dir <- "~/Downloads/aclImdb"
train_dir <- file.path(imdb_dir, "train")
labels <- c()
texts <- c()
for (label_type in c("neg", "pos")) {
  label <- switch(label_type, neg = 0, pos = 1)
  dir_name <- file.path(train_dir, label_type)
  for (fname in list.files(dir_name, pattern = glob2rx("*.txt"),
                           full.names = TRUE)) {
    texts <- c(texts, readChar(fname, file.info(fname)$size))
    labels <- c(labels, label)
  }
}

# Listing 6.9. Tokenizing the text of the raw IMDB data ####

maxlen <- 100                 # We will cut reviews after 100 words
training_samples <- 200       # We will be training on 200 samples
validation_samples <- 10000   # We will be validating on 10000 samples
max_words <- 10000            # We will only consider the top 10,000 words in the dataset
tokenizer <- text_tokenizer(num_words = max_words) %>%
  fit_text_tokenizer(texts)
sequences <- texts_to_sequences(tokenizer, texts)
word_index = tokenizer$word_index
cat("Found", length(word_index), "unique tokens.\n")



data <- pad_sequences(sequences, maxlen = maxlen)
labels <- as.array(labels)
cat("Shape of data tensor:", dim(data), "\n")

cat('Shape of label tensor:', dim(labels), "\n")



# Split the data into a training set and a validation set
# But first, shuffle the data, since we started from data
# where sample are ordered (all negative first, then all positive).
indices <- sample(1:nrow(data))
training_indices <- indices[1:training_samples]
validation_indices <- indices[(training_samples + 1):
                                (training_samples + validation_samples)]
x_train <- data[training_indices,]
y_train <- labels[training_indices]
x_val <- data[validation_indices,]
y_val <- labels[validation_indices]



# Download the GloVe word embeddings ####



# Listing 6.10. Parsing the GloVe word-embeddings file ####

# Listing 6.11. Preparing the GloVe word-embeddings matrix

# Listing 6.12. Model definition

# Listing 6.13. Loading pretrained word embeddings into the embedding layer

# Listing 6.14. Training and evaluation

model %>% compile(
  optimizer = "rmsprop",
  loss = "binary_crossentropy",
  metrics = c("acc")
)
history <- model %>% fit(
  x_train, y_train,
  epochs = 20,
  batch_size = 32,
  validation_data = list(x_val, y_val)
)
save_model_weights_hdf5(model, "pre_trained_glove_model.h5")



# Listing 6.17. Tokenizing the data of the test set ####

test_dir <- file.path(imdb_dir, "test")
labels <- c()
texts <- c()
for (label_type in c("neg", "pos")) {
  label <- switch(label_type, neg = 0, pos = 1)
  dir_name <- file.path(test_dir, label_type)
  for (fname in list.files(dir_name, pattern = glob2rx("*.txt"),
                           full.names = TRUE)) {
    texts <- c(texts, readChar(fname, file.info(fname)$size))
    labels <- c(labels, label)
  }
}
sequences <- texts_to_sequences(tokenizer, texts)
x_test <- pad_sequences(sequences, maxlen = maxlen)
y_test <- as.array(labels)



# Listing 6.18. Evaluating the model on the test set

model %>%
  load_model_weights_hdf5("pre_trained_glove_model.h5") %>%
  evaluate(x_test, y_test)



## >> 6.2-understanding-recurrent-neural-networks.nb.html ####

library(keras)
timesteps <- 10 # 100
input_features <- 4 # 32
output_features <- 8 # 64
random_array <- function(dim) {
  array(runif(prod(dim)), dim = dim)
}
inputs <- random_array(dim = c(timesteps, input_features))
state_t <- rep_len(0, length = c(output_features))
W <- random_array(dim = c(output_features, input_features))
U <- random_array(dim = c(output_features, output_features))
b <- random_array(dim = c(output_features, 1))
output_sequence <- array(0, dim = c(timesteps, output_features))
for (i in 1:nrow(inputs)) {
  input_t <- inputs[i,]
  output_t <- tanh(as.numeric((W %*% input_t) + (U %*% state_t) + b))
  output_sequence[i,] <- as.numeric(output_t)
  state_t <- output_t
}



layer_simple_rnn(units = 32)


library(keras)
model <- keras_model_sequential() %>%
  layer_embedding(input_dim = 10000, output_dim = 32) %>%
  layer_simple_rnn(units = 32, return_sequences = TRUE) %>%
  layer_simple_rnn(units = 32)  # This last layer only returns the last outputs.
summary(model)





# . layer_simple_rnn -> layer_lstm -> layer_gru ####

# . . Applying to IMDB data: layer_embedding %>% layer_lstm ####

if (F) {
  # FROM BOOK:
  #
  # model <- keras_model_sequential() %>%
  #   layer_embedding(input_dim = max_features, output_dim = 32) %>%
  #   layer_lstm(units = 32) %>%
  #   layer_dense(units = 1, activation = "sigmoid")
  # model %>% compile(
  #   optimizer = "rmsprop",
  #   loss = "binary_crossentropy",
  #   metrics = c("acc")
  # )
  # history <- model %>% fit(
  #   input_train, y_train,
  #   epochs = 10,
  #   batch_size = 128,
  #   validation_split = 0.2
  # )
  #
  #
  # FROM https://tensorflow.rstudio.com/blog/time-series-forecasting-with-recurrent-neural-networks.html
  #

# Listing 6.22. Preparing the IMDB data #####

  # Number of words to consider as features
  max_features <- 10000

  # Cuts off texts after this number of words
  maxlen <- 500

  imdb <- dataset_imdb(num_words = max_features)
  c(c(x_train, y_train), c(x_test, y_test)) %<-% imdb

  # Reverses sequences
  x_train <- lapply(x_train, rev)
  x_test <- lapply(x_test, rev)

  # Pads sequences
  x_train <- pad_sequences(x_train, maxlen = maxlen)  <4>
    x_test <- pad_sequences(x_test, maxlen = maxlen)

# Listing 6.23. Training the model with embedding and simple RNN layers ####

  model <- keras_model_sequential() %>%
    layer_embedding(input_dim = max_features, output_dim = 32) %>%
    layer_simple_rnn(units = 32) %>%
    layer_dense(units = 1, activation = "sigmoid")
  model %>% compile(
    optimizer = "rmsprop",
    loss = "binary_crossentropy",
    metrics = c("acc")
  )
  history <- model %>% fit(
    input_train, y_train,
    epochs = 10,
    batch_size = 128,
    validation_split = 0.2
  )

# Listing 6.27. Using the LSTM layer in Keras ####

# . model4: + layer_embedding + layer_lstm (applied to text classification)####

  model4 <- keras_model_sequential() %>%
    layer_embedding(input_dim = max_features, output_dim = 32) %>%  # 128) %>%
    layer_lstm(units = 32) %>%
    layer_dense(units = 1, activation = "sigmoid")

  model4 %>% compile(
    optimizer = "rmsprop",
    loss = "binary_crossentropy",
    metrics = c("acc")
  )

  history <- model4 %>% fit(
    x_train, y_train,
    epochs = 10,
    batch_size = 128,
    validation_split = 0.2
  )

  # Listing 6.42 = 6.27, but using reverse sequences Training and evaluating an LSTM using reversed sequences ####
  library(keras)
  max_features <- 10000
  maxlen <- 500
  imdb <- dataset_imdb(num_words = max_features)
  c(c(x_train, y_train), c(x_test, y_test)) %<-% imdb
  x_train <- lapply(x_train, rev)
  x_test <- lapply(x_test, rev)
  x_train <- pad_sequences(x_train, maxlen = maxlen)
  x_test <- pad_sequences(x_test, maxlen = maxlen)

  model4 <- keras_model_sequential() %>%
    layer_embedding(input_dim = max_features, output_dim = 128) %>%
    layer_lstm(units = 32) %>%
    layer_dense(units = 1, activation = "sigmoid")
  model4 %>% compile(
    optimizer = "rmsprop",
    loss = "binary_crossentropy",
    metrics = c("acc")
  )
  history <- model4 %>% fit(
    x_train, y_train,
    epochs = 10,
    batch_size = 128,
    validation_split = 0.2
  )


  # Listing 6.43. Training and evaluating a bidirectional LSTM ####

# . model5:  bidirectional + lstm (applied to text classification) ####

  model5 <- keras_model_sequential() %>%
    layer_embedding(input_dim = max_features, output_dim = 32) %>%
    bidirectional(
      layer_lstm(units = 32)
    ) %>%
    layer_dense(units = 1, activation = "sigmoid")

  model5 %>% compile(
    optimizer = "rmsprop",
    loss = "binary_crossentropy",
    metrics = c("acc")
  )

  history <- model5 %>% fit(
    x_train, y_train,
    epochs = 10,
    batch_size = 128,
    validation_split = 0.2
  )


  # Listing 6.44. Training a bidirectional GRU ####
  # . model6: bidirectional + gru (applied to forecasting) ------------------------------------------------------------------


  model6 <- keras_model_sequential() %>%
    bidirectional(
      layer_gru(units = 32), input_shape = list(NULL, dim(data)[[-1]])
    ) %>%
    layer_dense(units = 1)

  model6 %>% compile(
    optimizer = optimizer_rmsprop(),
    loss = "mae"
  )

  history <- model6 %>% fit_generator(
    train_gen,
    steps_per_epoch = 500,
    epochs = 40,sweep
    validation_data = val_gen,
    validation_steps = val_steps
  )


  # > Plot RNN results - Ref: https://www.datacamp.com/community/tutorials/keras-r-deep-learning ----
  # Plot Accuracy results


  plot(history)

  # Plot the model loss of the training data
  plot(history$metrics$loss, main="Model Loss", xlab = "epoch", ylab="loss", col="blue", type="l")
  # Plot the model loss of the test data
  lines(history$metrics$val_loss, col="green")
  # Add legend
  legend("topright", c("train","test"), col=c("blue", "green"), lty=c(1,1))


  # Plot the accuracy of the training data
  plot(history$metrics$acc, main="Model Accuracy", xlab = "epoch", ylab="accuracy", col="blue", type="l")
  # Plot the accuracy of the validation data
  lines(history$metrics$val_acc, col="green")
  # Add Legend
  legend("bottomright", c("train","test"), col=c("blue", "green"), lty=c(1,1))





}


## > 6.4-sequence-processing-with-convnets.nb.html ####
##
##
##
# Listing 6.45. Preparing the IMDB data ####

library(keras)
max_features <- 10000
max_len <- 500
cat("Loading data...\n")
imdb <- dataset_imdb(num_words = max_features)
c(c(x_train, y_train), c(x_test, y_test)) %<-% imdb
cat(length(x_train), "train sequences\n")
cat(length(x_test), "test sequences")
cat("Pad sequences (samples x time)\n")
x_train <- pad_sequences(x_train, maxlen = max_len)
x_test <- pad_sequences(x_test, maxlen = max_len)
cat("x_train shape:", dim(x_train), "\n")
cat("x_test shape:", dim(x_test), "\n")



# Listing 6.46. Training and evaluating a simple 1D convnet on the IMDB data ####

model <- keras_model_sequential() %>%
  layer_embedding(input_dim = max_features, output_dim = 128,
                  input_length = max_len) %>%
  layer_conv_1d(filters = 32, kernel_size = 7, activation = "relu") %>%
  layer_max_pooling_1d(pool_size = 5) %>%
  layer_conv_1d(filters = 32, kernel_size = 7, activation = "relu") %>%
  layer_global_max_pooling_1d() %>%
  layer_dense(units = 1)
summary(model)
model %>% compile(
  optimizer = optimizer_rmsprop(lr = 1e-4),
  loss = "binary_crossentropy",
  metrics = c("acc")
)
history <- model %>% fit(
  x_train, y_train,
  epochs = 10,
  batch_size = 128,
  validation_split = 0.2
)


# >>> APPENDIX: D7 functions ###############################
#


# . Create your own text dataset - to play with ####


library(tidyverse)
library(data.table)

filename <- "c:\\Users\\Computer\\Documents\\GitHub\\_TEXT\\txtSng\\01-Two-Lives-book-excerpt.Rmd"
filename <- "c:\\Users\\Computer\\Documents\\GitHub\\_TEXT\\txtSng\\2012-flows.Rmd"

if (F) { # - Way #1 I used it in my StakeOverfloor question
  library(readtext)
  df <- readtext(filename);  dt <- df %>% data.table()
  strEntireText <- dt[1]$text
  # Creates one LOOOONG string: > names(dt)   [1] "doc_id" "text"
}


## We don't want to mark R comments as sections.
d7.detect_codeblocks <- function(text) {
  blocks <- text %>%
    str_detect("```") %>%
    cumsum()
  blocks %% 2 != 0
}

## A df where each line is a row in the rmd file.

text2 = read_lines(filename); raw <- data_frame(text = text2)
#dt2 <- text %>% data.table()

#raw1 <- raw %>% mutate (text = ifelse(str_length(text)==0),"\n", text)

dt <- raw %>% data.table()
dt[str_length(text)==0, text:="\n"]
dt[text=="\n"]

dt <-
  dt %>%
  mutate(
    code_block = d7.detect_codeblocks(text),
    section = text %>%
      str_match("^# .*") %>%
      str_remove("^#+ +"),
    section = ifelse(code_block, NA, section),
    subsection = text %>%
      str_match("^## .*") %>%
      str_remove("^#+ +"),
    subsection = ifelse(code_block, NA, subsection)
  ) %>%
  tidyr::fill(section, subsection)


#to glue the text together within sections/subsections,
## then just group by them and flatten the text.

dtChapters <- dt %>%
  group_by(section, subsection) %>%
  slice(-1) %>%                           # remove the header
  summarize(
    text = text %>%
      str_flatten(" ") %>%
      str_trim()
  ) %>%
  ungroup() %>%
  data.table()


for(i in 1:nrow(dtChapters)){
  cat(i)
  print(dtChapters[i]$section)
  print(dtChapters[i]$subsection)
  print(dtChapters[i]$text)
  if (readline("continue? ") == 'n') break
}

dtChapters[c(2) ]$text %>% str_c() %>% str_wrap()



# . Assign your text dataset to samples to be use in tutorial excercises above ####

samples <- dtChapters$text[-1]
samples <- samples[-1]

#



# >>> PLAYGROUND:  Related to: DLwR-s6.1-RNN-for-text.R ####

library(keras)

imdb <- dataset_imdb(num_words = 10000)

if (T) {
  c(c(train_data, train_labels), c(test_data, test_labels)) %<-% imdb
} else {
  #multi-assignment operator (%<-%) from the zeallot package to unpack the list into a set of distinct variables. This could equally be written as follows:
  imdb <- dataset_imdb(num_words = 10000)
  train_data <- imdb$train$x
  train_labels <- imdb$train$y
  test_data <- imdb$test$x
  test_labels <- imdb$test$y
}

library(keras)


nWORDS <-  1111 # 10000 Number Of highest frequency words to use for this project
imdb <- dataset_imdb(num_words = nWORDS)
train_data <- imdb$train$x
train_labels <- imdb$train$y
test_data <- imdb$test$x
test_labels <- imdb$test$y

# c(c(input_train, y_train), c(input_test, y_test)) %<-% imdb


if (F) { # to decode them
  # Named list mapping words to an integer index.
  word_index <- dataset_imdb_word_index()
  reverse_word_index <- names(word_index); names(reverse_word_index) <- word_index

  showReviewForID <- function(listCodedReview) {
    # Decodes the review. Note that the indices are offset by 3 because 0, 1, and
    # 2 are reserved indices for "padding," "start of sequence," and "unknown."
    decoded_review <- sapply(unlist(listCodedReview), function(index) {
      word <- if (index >= 3) reverse_word_index[[as.character(index - 3)]]
      if (!is.null(word)) word else "???"
    })
    cat(decoded_review)
  }

  showReviewForID(train_data[1])
}




# = this is my 1994 flood-fill technique !
vectorize_sequences <- function(sequences, dimension = nWORDS) {
  # Creates an all-zero matrix of shape (length(sequences), dimension)
  results <- matrix(0, nrow = length(sequences), ncol = dimension)
  for (i in 1:length(sequences))
    # Sets specific indices of results[i] to 1s
    results[i, sequences[[i]]] <- 1
  results
}

x_train <- vectorize_sequences(train_data)
x_test <- vectorize_sequences(test_data)
y_train <- as.numeric(train_labels)
y_test <- as.numeric(test_labels)


model <- keras_model_sequential() %>%
  layer_dense(units = 16, activation = "relu", input_shape = c(nWORDS)) %>%
  layer_dense(units = 16, activation = "relu") %>%  #"tanh"
  layer_dense(units = 1, activation = "sigmoid")

summary(model)

model %>% compile(
  optimizer = "rmsprop", # optimizer_rmsprop(lr=0.001), # "rmsprop",
  loss = "binary_crossentropy", # "mse" loss_binary_crossentropy
  metrics = c("accuracy")       # metric_binary_accuracy
)




val_indices <- 1:500
x_val <- x_train[val_indices,]
partial_x_train <- x_train[-val_indices,]

y_val <- y_train[val_indices]
partial_y_train <- y_train[-val_indices]


history <- model %>% fit(
  partial_x_train,
  partial_y_train,
  epochs = 5, # 20,
  batch_size = 512,
  validation_data = list(x_val, y_val)
)

# model %>% fit(
#   partial_x_train,
#   partial_y_train,
#   epochs = 5, # 20,
#   batch_size = 512,
#   validation_data = list(x_val, y_val)
# )

plot(history)

results <- model %>% evaluate(x_test, y_test)
results


model %>% predict(x_test[1:10,])
showReviewForID(test_data[1])

##lapply(test_data[1:2],showReviewForID)




