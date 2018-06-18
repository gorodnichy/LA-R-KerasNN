# https://livebook.manning.com/#!/book/deep-learning-with-r/chapter-7/1 #####
# >>> Chapter 7. Advanced deep-learning best practices ####
#
# Content:
# The Keras functional API
# Using Keras callbacks
# Working with the TensorBoard visualization tool
# Important best practices for developing state-of-the-art models

# References used in the text:
# https://arxiv.org/abs/1409.4842 - Going Deeper with Convolutions
# https://arxiv.org/abs/1512.03385 - Computer Science > Computer Vision and Pattern Recognition Deep Residual Learning for Image Recognition
# https://arxiv.org/abs/1312.4400 - Network In Network
# https://arxiv.org/abs/1610.02357 - Xception: Deep Learning with Depthwise Separable Convolutions
# https://arxiv.org/abs/1512.03385 - Deep Residual Learning for Image Recognition
# https://arxiv.org/abs/1502.03167 - Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift
# https://arxiv.org/abs/1702.03275 - Batch Renormalization: Towards Reducing Minibatch Dependence in Batch-Normalized Models
# https://arxiv.org/abs/1706.02515 - Self-Normalizing Neural Networks

# Other: Uncertainty in Deep Learning (PhD Thesis) http://mlg.eng.cam.ac.uk/yarin/blog_2248.html#thesis

# 7.1  The Keras functional API ####

library(keras)
seq_model <- keras_model_sequential() %>%                          1
layer_dense(units = 32, activation = "relu", input_shape = c(64)) %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_dense(units = 10, activation = "softmax")
input_tensor <- layer_input(shape = c(64))                            2
output_tensor <- input_tensor %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_dense(units = 10, activation = "softmax")
model <- keras_model(input_tensor, output_tensor)                     3
summary(model)




# Listing 7.1. Functional API implementation of a two-input question-answering model ####
#
library(keras)
text_vocabulary_size <- 10000
ques_vocabulary_size <- 10000
answer_vocabulary_size <- 500
text_input <- layer_input(shape = list(NULL),
                          dtype = "int32", name = "text")
encoded_text <- text_input %>%
  layer_embedding(input_dim = 64, output_dim = text_vocabulary_size) %>%
layer_lstm(units = 32)
question_input <- layer_input(shape = list(NULL),
                              dtype = "int32", name = "question")
encoded_question <- question_input %>%
  layer_embedding(input_dim = 32, output_dim = ques_vocabulary_size) %>%
  layer_lstm(units = 16)
concatenated <- layer_concatenate(list(encoded_text, encoded_question))
answer <- concatenated %>%
layer_dense(units = answer_vocabulary_size, activation = "softmax")
model <- keras_model(list(text_input, question_input), answer)
model %>% compile(
  optimizer = "rmsprop",
  loss = "categorical_crossentropy",
  metrics = c("acc")
)



# Listing 7.2. Feeding data to a multi-input model ####
num_samples <- 1000
max_length <- 100
random_matrix <- function(range, nrow, ncol) {
  matrix(sample(range, size = nrow * ncol, replace = TRUE),
         nrow = nrow, ncol = ncol)
}
text <- random_matrix(1:text_vocabulary_size, num_samples, max_length)
question <- random_matrix(1:ques_vocabulary_size, num_samples, max_length)
answers <- random_matrix(0:1, num_samples, answer_vocabulary_size)
model %>% fit(
  list(text, question), answers,
  epochs = 10, batch_size = 128
)
model %>% fit(
  list(text = text, question = question), answers,
  epochs = 10, batch_size = 128
)



# Listing 7.3. Functional API implementation of a three-output model ####


library(keras)
vocabulary_size <- 50000
num_income_groups <- 10
posts_input <- layer_input(shape = list(NULL),
                           dtype = "int32",  name = "posts")
embedded_posts <- posts_input %>%
  layer_embedding(input_dim = 256, output_dim = vocabulary_size)
base_model <- embedded_posts %>%
  layer_conv_1d(filters = 128, kernel_size = 5, activation = "relu") %>%
  layer_max_pooling_1d(pool_size = 5) %>%
  layer_conv_1d(filters = 256, kernel_size = 5, activation = "relu") %>%
  layer_conv_1d(filters = 256, kernel_size = 5, activation = "relu") %>%
  layer_max_pooling_1d(pool_size = 5) %>%
  layer_conv_1d(filters = 256, kernel_size = 5, activation = "relu") %>%
  layer_conv_1d(filters = 256, kernel_size = 5, activation = "relu") %>%
  layer_global_max_pooling_1d() %>%
  layer_dense(units = 128, activation = "relu")
age_prediction <- base_model %>%                                   1
layer_dense(units = 1, name = "age")
income_prediction <- base_model %>%
  layer_dense(num_income_groups, activation = "softmax", name = "income")
gender_prediction <- base_model %>%
  layer_dense(units = 1, activation = "sigmoid", name = "gender")
model <- keras_model(
  posts_input,
  list(age_prediction, income_prediction, gender_prediction)
)



#Listing 7.4. Compilation options of a multi-output model: multiple losses
model %>% compile(
  optimizer = "rmsprop",
  loss = c("mse", "categorical_crossentropy", "binary_crossentropy")
)
model %>% compile(
  optimizer = "rmsprop",
  loss = list(
    age = "mse",
    income = "categorical_crossentropy",
    gender = "binary_crossentropy"
  )
)


# Listing 7.5. Compilation options of a multi-output model: loss weighting ####
#
model %>% compile(
  optimizer = "rmsprop",
  loss = c("mse", "categorical_crossentropy", "binary_crossentropy"),
  loss_weights = c(0.25, 1, 10)
)
model %>% compile(
  optimizer = "rmsprop",
  loss = list(
    age = "mse",
    income = "categorical_crossentropy",
    gender = "binary_crossentropy"
  ),
  loss_weights = list(
    age = 0.25,
    income = 1,
    gender = 10
  )
)


# Listing 7.6. Feeding data to a multi-output model ####



model %>% fit(                                                      1
                                                                    posts, list(age_targets, income_targets, gender_targets),         1
                                                                    epochs = 10, batch_size = 64                                      1
)                                                                   1
model %>% fit(                                                      2
                                                                    posts, list(                                                      2
                                                                                                                                      age = age_targets,                                              2
                                                                                                                                      income = income_targets,                                        2
                                                                                                                                      gender = gender_targets                                         2
                                                                    ),                                                                2
                                                                    epochs = 10, batch_size = 64                                      2
)

# 7.1.4. Directed acyclic graphs of layers ####



# THE PURPOSE OF 1 × 1 CONVOLUTIONS - UNLISTED ####
library(keras)
branch_a <- input %>%
  layer_conv_2d(filters = 128, kernel_size = 1,
                activation = "relu", strides = 2)               1
branch_b <- input %>%
  layer_conv_2d(filters = 128, kernel_size = 1,
                activation = "relu") %>%
  layer_conv_2d(filters = 128, kernel_size = 3,
                activation = "relu", strides = 2)               2
branch_c <- input %>%
  layer_average_pooling_2d(pool_size = 3, strides = 2) %>%        3
layer_conv_2d(filters = 128, kernel_size = 3,
              activation = "relu")
branch_d <- input %>%
  layer_conv_2d(filters = 128, kernel_size = 1,
                activation = "relu") %>%
  layer_conv_2d(filters = 128, kernel_size = 3,
                activation = "relu") %>%
  layer_conv_2d(filters = 128, kernel_size = 3,
                activation = "relu", strides = 2)
output <- layer_concatenate(list(                                 4
                                                                  branch_a, branch_b, branch_c, branch_d                          4
))





# UNLISTED - Residual connections ####
output <- input %>%
layer_conv_2d(filters = 128, kernel_size = 3,
              activation = "relu", padding = "same") %>%
  layer_conv_2d(filters = 128, kernel_size = 3,
                activation = "relu", padding = "same") %>%
  layer_conv_2d(filters = 128, kernel_size = 3,
                activation = "relu", padding = "same")
output <- layer_add(list(output, input))



# UNLISTED ####
output <- input %>%
  layer_conv_2d(filters = 128, kernel_size = 3,
                activation = "relu", padding = "same") %>%
  layer_conv_2d(filters = 128, kernel_size = 3,
                activation = "relu", padding = "same") %>%
  layer_max_pooling_2d(pool_size = 2, strides = 2)
residual <- input %>%
  layer_conv_2d(filters = 128, kernel_size = 1,
                strides = 2, padding = "same")
output <- layer_add(list(output, residual))




# 7.1.5. Layer weight sharing ####

# UNLISTED ####
library(keras)
lstm <- layer_lstm(units = 32)                                1
left_input <- layer_input(shape = list(NULL, 128))                2
left_output <- left_input %>% lstm()                              2
right_input <- layer_input(shape = list(NULL, 128))               3
right_output <- right_input %>% lstm()                            3
merged <- layer_concatenate(list(left_output, right_output))
predictions <- merged %>%                                         4
layer_dense(units = 1, activation = "sigmoid")                  4
model <- keras_model(list(left_input, right_input), predictions)  5
model %>% fit(                                                    5
                                                                  list(left_data, right_data), targets)                           5
)


# 7.1.6. Models as layers ####
#
# Importantly, in the functional API, models can be used as you’d use layers—effectively,
# you can think of a model as a “bigger layer.”
# This is true of models created with both the keras_model and keras_model_sequential functions.
# This means you can call a model on an input tensor and retrieve an output tensor:


y <- model(x)

c(y1, y2) %<-% <- model(list(x1, x2))


library(keras)
xception_base <- application_xception(weights = NULL,          1
                                      include_top = FALSE)     1
left_input <- layer_input(shape = c(250, 250, 3))              2
right_input <- layer_input(shape = c(250, 250, 3))             2
left_features = left_input %>% xception_base()                 3
right_features <- right_input %>% xception_base()              3
merged_features <- layer_concatenate(                          4
                                                               list(left_features, right_features)                          4
)




# 7.2. Inspecting and monitoring deep-learning models using Keras callba- acks and TensorBoard ####
# In this section, we’ll review ways to gain greater access to and control over what goes on inside your model during training. Launching a training run on a large dataset for tens of epochs using fit() or fit_generator() can be a bit like launching a paper airplane: past the initial impulse, you don’t have any control over its trajectory or its landing spot. If you want to avoid bad outcomes (and thus wasted paper airplanes), it’s smarter to use not a paper plane, but a drone that can sense its environment, send data back to its operator, and automatically make steering decisions based on its current state. The techniques we present here will transform the call to fit() from a paper airplane into a smart, autonomous drone that can self-introspect and dynamically take action.

# 7.2.1. Using callbacks to act on a model during trainin  ####
# When you’re training a model, there are many things you can’t predict from the start.
# In particular, you can’t tell how many epochs will be needed to get to an optimal validation loss.
# The examples so far have adopted the strategy of training for enough epochs that you begin
# overfitting, using the first run to figure out the proper number of epochs to train for,
# and then finally launching a new training run from scratch using this optimal number.
# Of course, this approach is wasteful.

# Here are some examples of ways you can use callbacks: ####

callback_model_checkpoint()
callback_early_stopping()
callback_learning_rate_scheduler()
callback_reduce_lr_on_plateau()
callback_csv_logger()

# UNLISTED - The model-checkpoint and early-stopping callbacks ####
library(keras)
callbacks_list <- list(                        1
                                               callback_early_stopping(                     2
                                                                                            monitor = "acc",                           3
                                                                                            patience = 1                               4
                                               ),
                                               callback_model_checkpoint(                   5
                                                                                            filepath = "my_model.h5",                  6
                                                                                            monitor = "val_loss",                      7
                                                                                            save_best_only = TRUE
                                               )
)
model %>% compile(
  optimizer = "rmsprop",
  loss = "binary_crossentropy",
  metrics = c("acc")                        8
)
model %>% fit(                                 9
                                               x, y,                                        9
                                               epochs = 10,                                 9
                                               batch_size = 32,                             9
                                               callbacks = callbacks_list,                  9
                                               validation_data = list(x_val, y_val)         9
)                                              9



# UNLISTED - The reduce-learning-rate-on-plateau callback ####

callbacks_list <- list(
  callback_reduce_lr_on_plateau(
    monitor = "val_loss",                    1
    factor = 0.1,                            2
    patience = 10                            3
  )
)
model %>% fit(                               4
                                             x, y,                                      4
                                             epochs = 10,                               4
                                             batch_size = 32,                           4
                                             callbacks = callbacks_list,                4
                                             validation_data = list(x_val, y_val)       4
)



# UNLISTED - Writing your own callback #####

on_epoch_begin          1
on_epoch_end            2
on_batch_begin          3
on_batch_end            4
on_train_begin          5
on_train_end            6


# UNLISTED - ####
library(keras)
library(R6)
LossHistory <- R6Class("LossHistory",
                       inherit = KerasCallback,
                       public = list(
                         losses = NULL,
                         on_batch_end = function(batch, logs = list()) {            1
                           self$losses <- c(self$losses, logs[["loss"]])            2
                         }
                       ))
history <- LossHistory$new()                                   3
model %>% fit(
  x, y,
  batch_size = 128,
  epochs = 20,
  callbacks = list(history)                                    4
)
> str(history$losses)                                          5
num [1:160] 0.634 0.615 0.631 0.664 0.626 ...



# 7.2.2. Introduction to TensorBoard: the TensorFlow visualization framework ####
# To do good research or develop good models, you need rich, frequent feedback about what’s going on inside your models during your experiments. That’s the point of running experiments: to get information about how well a model performs—as much information as possible. Making progress is an iterative process, or loop: you start with an idea and express it as an experiment, attempting to validate or invalidate your idea. You run this experiment and process the information it generates. This inspires your next idea. The more iterations of this loop you’re able to run, the more refined and powerful your ideas become. Keras helps you go from idea to experiment in the least possible time, and fast GPUs can help you get from experiment to result as quickly as possible. But what about processing the experiment results? That’s where Tensor-Board comes in.
#
#
#
#



# Listing 7.7. Text-classification model to use with TensorBoard ####
library(keras)
max_features <- 2000                                     1
max_len <- 500                                           2
imdb <- dataset_imdb(num_words = max_features)
c(c(x_train, y_train), c(x_test, y_test)) %<-% imdb
x_train <- pad_sequences(x_train, maxlen = max_len)
x_test = pad_sequences(x_test, maxlen = max_len)
model <- keras_model_sequential() %>%
  layer_embedding(input_dim = max_features, output_dim = 128,
                  input_length = max_len, name = "embed") %>%
  layer_conv_1d(filters = 32, kernel_size = 7, activation = "relu") %>%
  layer_max_pooling_1d(pool_size = 5) %>%
  layer_conv_1d(filters = 32, kernel_size = 7, activation = "relu") %>%
  layer_global_max_pooling_1d() %>%
  layer_dense(units = 1)
summary(model)
model %>% compile(
  optimizer = "rmsprop",
  loss = "binary_crossentropy",
  metrics = c("acc")
)

# Listing 7.8. Creating a directory for TensorBoard log files ####
> dir.create("my_log_dir")
#



# Listing 7.9. Training the model with a TensorBoard callback ####
tensorboard("my_log_dir")             1
callbacks = list(
  callback_tensorboard(
    log_dir = "my_log_dir",
    histogram_freq = 1,               2
    embeddings_freq = 1,              3
  )
)
history <- model %>% fit(
  x_train, y_train,
  epochs = 20,
  batch_size = 128,
  validation_split = 0.2,
  callbacks = callbacks
)



# 7.3. Getting the most out of your models ####


# Trying out architectures blindly works well enough if you just need something that works okay. In this section, we’ll go beyond “works okay” to “works great and wins machine-learning competitions” by offering you a quick guide to a set of must-know techniques for building state-of-the-art deep-learning models.

# 7.3.1. Advanced architecture patterns ####
# We covered one important design pattern in detail in the previous section: residual connections.
# There are two more design patterns you should know about: normalization and depthwise separable
# convolution. These patterns are especially relevant when you’re building high-performing deep convnets,
# but they’re commonly found in many other types of architectures as well.

# Batch normalization


mean <- apply(train_data, 2, mean)
std <- apply(train_data, 2, sd)
train_data <- scale(train_data, center = mean, scale = std)
test_data <- scale(test_data, center = mean, scale = std)



layer_conv_2d(filters = 32, kernel_size = 3, activation = "relu") %>%
  layer_batch_normalization()
layer_dense(units = 32, activation = "relu") %>%
  layer_batch_normalization()



# 7.3.2. Hyperparameter optimization
# When building a deep-learning model, you have to make many seemingly arbitrary decisions: How many layers should you stack? How many units or filters should go in each layer? Should you use relu as activation, or a different function? Should you use layer_batch_normalization after a given layer? How much dropout should you use? And so on. These architecture-level parameters are called hyperparameters to distinguish them from the parameters of a model, which are trained via backpropagation.





# >>>> APPENDIX: library(udpipe) etc. #####

library(udpipe)
udmodel <- udpipe_download_model(language = "russian")




# > https://rpubs.com/pjmurphy/265713 ####
lirbary(fpc)
