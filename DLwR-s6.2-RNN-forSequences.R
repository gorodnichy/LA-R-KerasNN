# DLwR-s6.1-RNN-forSequences.R
# >>>> Chapter 6. Deep learning for text and sequences (Part 2: RNN, [LSTM], GRU - for jena weather data) ####
# https://livebook.manning.com/#!/book/deep-learning-with-r/chapter-6
#
# >>> Used for https://tensorflow.rstudio.com/blog/time-series-forecasting-with-recurrent-neural-networks.html ####

# See also: Electricity load forecasting with LSTM (python)
# https://github.com/dafrie/lstm-load-forecasting/tree/master/lstm_load_forecasting

## >> 6.3-advanced-usage-of-recurrent-neural-networks.nb.html #####
#
#
# In this section, we’ll review three advanced techniques for improving the performance
# and generalization power of recurrent neural networks.
#
# Until now, the only sequence data we’ve covered has been text data, such as the IMDB dataset and the Reuters dataset.
# In this section, you’ll play with a weather timeseries dataset.

# . . Applying to weather timeseries data: ####

sequence_generator <- function(start) {
  value <- start - 1
  function() {
    value <<- value + 1
    value
  }
}
gen <- sequence_generator(10)
gen()


library(keras)
if (F) install_keras()
# > Using r-tensorflow conda environment for TensorFlow installation

#if (F) { # this is done automaticaly with install_keras()
# library(tensorflow)
# install_tensorflow()
#}

if (F) {
  dir.create("jena_climate", recursive = TRUE)
  download.file(
    "https://s3.amazonaws.com/keras-datasets/jena_climate_2009_2016.csv.zip",
    "jena_climate/jena_climate_2009_2016.csv.zip"
  )
  unzip(
    "jena_climate/jena_climate_2009_2016.csv.zip",
    exdir = "jena_climate"
  )
}

# Listing 6.28. Inspecting the data of the Jena weather dataset ####

data_dir <- "jena_climate"
fname <- file.path(data_dir, "jena_climate_2009_2016.csv")
data <- read_csv(fname)

glimpse(data); #str(data)
summary(data);
names(data)
data %>% data.table %>% print(7)




#data is recorded every 10 minutes, you get 144 data points per day. For two weeks:
ggplot(data[1:(144*14),], aes(x = 1:(144*14), y = `T (degC)`)) + geom_line()

data <- data.matrix(data[,-1])

#Preparing data

train_data <- data[1:200000,] #data[1:200000,]
mean <- apply(train_data, 2, mean)
std <- apply(train_data, 2, sd)
data <- scale(data, center = mean, scale = std)


# data — The original array of floating-point data, which you normalized in listing 6.32.
# lookback — How many timesteps back the input data should go.
# delay — How many timesteps in the future the target should be.
# min_index and max_index — Indices in the data array that delimit which timesteps to draw from. This is useful for keeping a segment of the data for validation and another for testing.
# shuffle — Whether to shuffle the samples or draw them in chronological order.
# batch_size — The number of samples per batch.
# step — The period, in timesteps, at which you sample data. You’ll set it 6 in order to draw one data point every hour.

generator <- function(data, lookback, delay, min_index, max_index,
                      shuffle = FALSE, batch_size = 128, step = 6) {
  if (is.null(max_index))
    max_index <- nrow(data) - delay - 1
  i <- min_index + lookback

  function() {
    if (shuffle) {
      rows <- sample(c((min_index+lookback):max_index), size = batch_size)
    } else {
      if (i + batch_size >= max_index)
        i <<- min_index + lookback
      rows <- c(i:min(i+batch_size-1, max_index))
      i <<- i + length(rows)
    }

    samples <- array(0, dim = c(length(rows),
                                lookback / step,
                                dim(data)[[-1]]))
    targets <- array(0, dim = c(length(rows)))

    for (j in 1:length(rows)) {
      indices <- seq(rows[[j]] - lookback, rows[[j]]-1,
                     length.out = dim(samples)[[2]])
      samples[j,,] <- data[indices,]
      targets[[j]] <- data[rows[[j]] + delay,2]
    }

    list(samples, targets)
  }
}

# Listing 6.34. Preparing the training, validation, and test generators ####

lookback <- 1440 #last ten days: 10*144
step <- 6
delay <- 144 #one day in the future
batch_size <- 128

train_gen <- generator(
  data,
  lookback = lookback,
  delay = delay,
  min_index = 1,
  max_index = 200000,
  shuffle = TRUE,
  step = step,
  batch_size = batch_size
)

val_gen = generator(
  data,
  lookback = lookback,
  delay = delay,
  min_index = 200001,
  max_index = 300000,
  step = step,
  batch_size = batch_size
)

test_gen <- generator(
  data,
  lookback = lookback,
  delay = delay,
  min_index = 300001,
  max_index = NULL,
  step = step,
  batch_size = batch_size
)

# How many steps to draw from val_gen in order to see the entire validation set
val_steps <- (300000 - 200001 - lookback) / batch_size

# How many steps to draw from test_gen in order to see the entire test set
test_steps <- (nrow(data) - 300001 - lookback) / batch_size



if (F) {
# Listing 6.35. Computing the common-sense baseline MAE ####
  evaluate_naive_method <- function() {
    batch_maes <- c()
    for (step in 1:val_steps) {
      c(samples, targets) %<-% val_gen()
      preds <- samples[,dim(samples)[[2]],2]
      mae <- mean(abs(preds - targets))
      batch_maes <- c(batch_maes, mae)
    }
    print(mean(batch_maes))
  }

  celsius_mae <- evaluate_naive_method() *  std[[2]] # [1] 8.852521
  # Error = celsius_mae= 0.2789863 %


}





# Listing 6.37. Training and evaluating a densely connected model ####
# . model0: layer_flatten %>% layer_dense ####


model0 <- keras_model_sequential() %>%
  layer_flatten(input_shape = c(lookback / step, ncol(data))) %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_dense(units = 1)

model0 %>% compile(
  optimizer = optimizer_rmsprop(),
  loss = "mae"
)

history <- model0 %>% fit_generator(
  train_gen,
  steps_per_epoch = 5, # 500,
  epochs = 3, # 20,
  validation_data = val_gen,
  validation_steps = val_steps
)


# . model1: layer_gru() <- layer_flatten()  ------------------------------------------------------------------



model1 <- keras_model_sequential() %>%
  layer_gru(units = 32,
            input_shape = list(NULL, dim(data)[[-1]])) %>% # ?? ->remove first (-1) list item, which is timestamp
  layer_dense(units = 1)

model1 %>% compile(
  optimizer = optimizer_rmsprop(),
  loss = "mae"
)


history <- model1 %>% fit_generator(
  train_gen,
  steps_per_epoch = 500,
  epochs = 20,
  validation_data = val_gen,
  validation_steps = val_steps
)



# . model2: + dropout = 0.2, recurrent_dropout = 0.2,  ------------------------------------------------------------------


model2 <- keras_model_sequential() %>%
  layer_gru(units = 32, dropout = 0.2, recurrent_dropout = 0.2,
            input_shape = list(NULL, dim(data)[[-1]])) %>%
  layer_dense(units = 1)

model2 %>% compile(
  optimizer = optimizer_rmsprop(),
  loss = "mae"
)

history <- model2 %>% fit_generator(
  train_gen,
  steps_per_epoch = 500,
  epochs = 40,
  validation_data = val_gen,
  validation_steps = val_steps
)


# . model3: + stacked GRU  ------------------------------------------------------------------
# Listing 6.41. Training and evaluating a dropout-regularized, stacked GRU model ####

model3 <- keras_model_sequential() %>%
  layer_gru(units = 32,
            dropout = 0.1,
            recurrent_dropout = 0.5,
            return_sequences = TRUE,
            input_shape = list(NULL, dim(data)[[-1]])) %>%
  layer_gru(units = 64, activation = "relu",
            dropout = 0.1,
            recurrent_dropout = 0.5) %>%
  layer_dense(units = 1)

model3 %>% compile(
  optimizer = optimizer_rmsprop(),
  loss = "mae"
)

history <- model3 %>% fit_generator(
  train_gen,
  steps_per_epoch = 500,
  epochs = 40,
  validation_data = val_gen,
  validation_steps = val_steps
)



# . model4: increase capacity (gru layers)  ------------------------------------------------------------------



model <- keras_model_sequential() %>%
  layer_gru(units = 32,
            dropout = 0.1,
            recurrent_dropout = 0.5,
            return_sequences = TRUE,
            input_shape = list(NULL, dim(data)[[-1]])) %>%
  layer_gru(units = 64, activation = "relu",
            dropout = 0.1,
            recurrent_dropout = 0.5) %>%
  layer_dense(units = 1)
model %>% compile(
  optimizer = optimizer_rmsprop(),
  loss = "mae"
)
history <- model %>% fit_generator(
  train_gen,
  steps_per_epoch = 500,
  epochs = 40,
  validation_data = val_gen,
  validation_steps = val_steps
)




#########################################################
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


#Listing 6.47. Training and evaluating a simple 1D convnet on the Jena data ####
model <- keras_model_sequential() %>%
  layer_conv_1d(filters = 32, kernel_size = 5, activation = "relu",
                input_shape = list(NULL, dim(data)[[-1]])) %>%
  layer_max_pooling_1d(pool_size = 3) %>%
  layer_conv_1d(filters = 32, kernel_size = 5, activation = "relu") %>%
  layer_max_pooling_1d(pool_size = 3) %>%
  layer_conv_1d(filters = 32, kernel_size = 5, activation = "relu") %>%
  layer_global_max_pooling_1d() %>%
  layer_dense(units = 1)
model %>% compile(
  optimizer = optimizer_rmsprop(),
  loss = "mae"
)
history <- model %>% fit_generator(
  train_gen,
  steps_per_epoch = 500,
  epochs = 20,
  validation_data = val_gen,
  validation_steps = val_steps
)

# Listing 6.48. Preparing higher-resolution data generators for the Jena dataset ####
step <- 3                        1
lookback <- 720
delay <- 144
train_gen <- generator(
  data,
  lookback = lookback,
  delay = delay,
  min_index = 1,
  max_index = 200000,
  shuffle = TRUE,
  step = step
)
val_gen <- generator(
  data,
  lookback = lookback,
  delay = delay,
  min_index = 200001,
  max_index = 300000,
  step = step
)
test_gen <- generator(
  data,
  lookback = lookback,
  delay = delay,
  min_index = 300001,
  max_index = NULL,
  step = step
)
val_steps <- (300000 - 200001 - lookback) / 128
test_steps <- (nrow(data) - 300001 - lookback) / 128


# Listing 6.49. Model combining a 1D convolutional base and a GRU layer ####
model <- keras_model_sequential() %>%
  layer_conv_1d(filters = 32, kernel_size = 5, activation = "relu",
                input_shape = list(NULL, dim(data)[[-1]])) %>%
  layer_max_pooling_1d(pool_size = 3) %>%
  layer_conv_1d(filters = 32, kernel_size = 5, activation = "relu") %>%
  layer_gru(units = 32, dropout = 0.1, recurrent_dropout = 0.5) %>%
  layer_dense(units = 1)
summary(model)
model %>% compile(
  optimizer = optimizer_rmsprop(),
  loss = "mae"
)
history <- model %>% fit_generator(
  train_gen,
  steps_per_epoch = 500,
  epochs = 20,
  validation_data = val_gen,
  validation_steps = val_steps
)


#Listing 6.47. Training and evaluating a simple 1D convnet on the Jena data ####
model <- keras_model_sequential() %>%
  layer_conv_1d(filters = 32, kernel_size = 5, activation = "relu",
                input_shape = list(NULL, dim(data)[[-1]])) %>%
  layer_max_pooling_1d(pool_size = 3) %>%
  layer_conv_1d(filters = 32, kernel_size = 5, activation = "relu") %>%
  layer_max_pooling_1d(pool_size = 3) %>%
  layer_conv_1d(filters = 32, kernel_size = 5, activation = "relu") %>%
  layer_global_max_pooling_1d() %>%
  layer_dense(units = 1)
model %>% compile(
  optimizer = optimizer_rmsprop(),
  loss = "mae"
)
history <- model %>% fit_generator(
  train_gen,
  steps_per_epoch = 500,
  epochs = 20,
  validation_data = val_gen,
  validation_steps = val_steps
)

# Listing 6.48. Preparing higher-resolution data generators for the Jena dataset ####
step <- 3
lookback <- 720
delay <- 144
train_gen <- generator(
  data,
  lookback = lookback,
  delay = delay,
  min_index = 1,
  max_index = 200000,
  shuffle = TRUE,
  step = step
)
val_gen <- generator(
  data,
  lookback = lookback,
  delay = delay,
  min_index = 200001,
  max_index = 300000,
  step = step
)
test_gen <- generator(
  data,
  lookback = lookback,
  delay = delay,
  min_index = 300001,
  max_index = NULL,
  step = step
)
val_steps <- (300000 - 200001 - lookback) / 128
test_steps <- (nrow(data) - 300001 - lookback) / 128


# Listing 6.49. Model combining a 1D convolutional base and a GRU layer ####
model <- keras_model_sequential() %>%
  layer_conv_1d(filters = 32, kernel_size = 5, activation = "relu",
                input_shape = list(NULL, dim(data)[[-1]])) %>%
  layer_max_pooling_1d(pool_size = 3) %>%
  layer_conv_1d(filters = 32, kernel_size = 5, activation = "relu") %>%
  layer_gru(units = 32, dropout = 0.1, recurrent_dropout = 0.5) %>%
  layer_dense(units = 1)
summary(model)
model %>% compile(
  optimizer = optimizer_rmsprop(),
  loss = "mae"
)
history <- model %>% fit_generator(
  train_gen,
  steps_per_epoch = 500,
  epochs = 20,
  validation_data = val_gen,
  validation_steps = val_steps
)



# APPENDIX: D7 functions ###############################
#



# . Create your time-series dataset - to play with ####


# BWT: Open Government -> [Historical Border Wait Times](https://open.canada.ca/data/en/dataset/000fe5aa-1d77-42d1-bfe7-458c51dacfef): ----

library(data.table);library(ggplot2); library(lubridate);
library(magrittr); library(stringr);  options(datatable.print.class=TRUE)

# downloaded 5.1 MB
dt <- fread("http://cbsa-asfc.gc.ca/data/bwt-taf-2016-07-01--2016-09-30-en.csv")

# names(dt); dt
# as.ordered(dt[[2]]) %>% levels; as.ordered(dt[[4]]) %>% levels;
# dt[str_detect(Location, "BC")]$Location %>% unique()

dt <- dt[str_detect(Location, "BC")] # select smaller subset - data from BC only
dt[, Updated := as.POSIXct(Updated, format = "%Y-%m-%d %H:%M ")  ]
dt$BWT <- str_replace(dt$`Travellers Flow`, "No delay", "0") %>%
  str_replace("Closed", "0") %>%   str_replace("Not applicable", "0") %>%
  str_replace("Missed entry", "NA") %>% as.numeric()

dt <- dt[, .(Updated, BWT, Location)]
plot(as.ts(dt))
dt <- dt[, ':='(hh=hour(Updated), wd=wday(Updated))]
dt

# plot(as.ts(dt))
# ggTimeSeries::ggplot_calendar_heatmap( dt, 'Updated', 'BWT')


theme_set(theme_minimal())
ggplot(dt[Updated>=dmy("01/09/2016")], aes(Updated,BWT, col=Location)) +
  geom_step() + facet_grid(Location ~ .)

#impute NA
nNA <- which(is.na(dt$BWT))
dt[nNA]$BWT <- dt[nNA+1]$BWT
dt[BWT==0, BWT:=1]

#Simplest possible: one-liner "solutions"
#model <- arima(as.ts(dt[Location=='Delta, BC',BWT]))
# library(GMDH)
# predictedBWT <- GMDH::fcast(as.ts(dt[Location=='Delta, BC',BWT]))
# plot(as.ts(predictedBWT))




# Car volumes: http://www.cascadegatewaydata.com/Reports  (Car Volumes + BWT on the Canada-US border )  ----
# Southbound Monthly Car Volumes,  	1/1/2008 	12/31/2014  - 5Kb
dt <- fread("c:/Users/Computer/Documents/_CODES/_OPEN/LA-R-Keras/dataSeqs/Traffic/query-volume-3ports.csv")
# 2008-2013 Lynden/Aldergrove monthly volumes 	1/1/2008 	12/31/2013 - 3kb
dt <- fread("c:/Users/Computer/Documents/_CODES/_OPEN/LA-R-Keras/dataSeqs/Traffic/query-volume.csv")
#2013 Average Weekend Delay, Peace Arch/Douglas 	1/1/2013 	12/31/2013 89Kb
dt <- fread("c:/Users/Computer/Documents/_CODES/_OPEN/LA-R-Keras/dataSeqs/Traffic/query-delay1.csv")



setnames(dtVolume, "Group Starts", "TIME")
dtVolume[, TIME := as.POSIXct(TIME, format = "%Y-%m-%d %H:%M ") ]
strCol = 1;   dtVolume[[strCol]] = as.ordered(dtVolume[[strCol]])

# plots <- list();
# for(i in 2:ncol(dt)) {
#   strX = names(dt)[i]
#   print(sprintf("%i: strX = %s", i, strX))
#   plots[[i]] <- ggplot(dt) + xlab(strX) +
#     geom_point(aes_string(strX),stat="count",size=2) +
#     geom_line(aes_string(strX),stat="count",size=1, col="grey")
#   print(plots[[i]])
# }
multiplot(plotlist = plots, cols = 1)

g <- list()
for(i in 1:(ncol(dt)-1) {
  strX = names(dt)[i+1]
  g[[i]] <- ggplot(dt, aes(y=TIME)) + labs(title=strX) +
    geom_line(aes_string(strX), col=i) +
    geom_label(data=data.frame(x=0,y=0), aes(x,y), col=i, label=strX, hjust=0, vjust=0)
}
GGally::ggmatrix(nrow=3, ncol=1, title = "Your data")

ggplot() + geom_line(aes(x=dt$`Sum - Volume (Lynden/Aldergrove North Cars)`), y=dt$TIME))

  dtDelay <- fread("c:/Users/Computer/Documents/_CODES/_OPEN/LA-R-Keras/dataSeqs/Traffic/query-delay1.csv") #on week-ends

line(dt[[1]], dt[[2]])

g <- list()
for(i in 1:(ncol(dt)-1)) {
  g[[i]] <- ggplot() + labs(y=names(dt)[i+1], x=names(dt)[1]) +
    geom_line(aes(y=dt[[i+1]], x=dt[[1]])) + geom_point(aes(y=dt[[i+1]], x=dt[[1]]))
  # geom_label(data=data.frame(x=0,y=0), aes(x,y), col=i, label=names(dt)[i+1], hjust=0, vjust=0)
  print(g[[i]])
}
GGally::ggmatrix(g, nrow=ncol(dt)-1, ncol=1, title = "Your data")

ggfortify::autoplot(g)


# currency exchange data ----

dtExchange= fread ("dataCBSA/currencyCAN_US_2008-2018.csv") #Exchange Rates 20070124-20170124.csv");
setnames(dtExchange, c("TIME","USD"))
dtExchange = dtExchange[,TIME := as.POSIXct(as.Date(dmy(TIME)))]
dt = dtExchange[dt,roll=Inf,on="TIME"]


