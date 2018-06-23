# LA-R-KerasNN

### *'Learn and Apply Artificial Neural Networks (aka Deep Learning) with Keras/Tensorflow in R/RStudio, Efficiently'*, <br> Using complete prewritten codes and your own data, your own data, and data.table package. 

Based on:
- https://github.com/jjallaire/deep-learning-with-r-notebooks
- https://keras.rstudio.com (= https://tensorflow.rstudio.com/keras)
- https://www.manning.com/books/deep-learning-with-r
- everything else found useful on the Web, 
- and [my own work on ANN back from 1995-2005 on PINN](https://sites.google.com/site/dmitrygorodnichy/ANN/PhD-PINN) :)


Notes:
- Latest version of RStudio is always recommended (Presently, Version 1.1.447 – 2018 )
- All codes are retrieved from original sources, simplified, merged and directly runnable from RStudio

- The order of lessons is recommended by indices: e.g. 1-1-1 goes prior to 1-3-1. [.] are optional.
- `# .... ####` comments are used for quick navigation from one example/section to another within RStudio IDE.
- `# >>> ... <<< ####` indicate Main sections
- Data to play with (traffic, favourite readings) are provided, inc. very small sets to run fast.

See also:
- ['Learn and Apply interactive programming and presentations with RStudio Shiny'](https://github.com/gorodnichy/LA-R-Rmd-Shiny).
- ['Learn and Apply Text Analysis in R, Efficiently'](https://github.com/gorodnichy/LA-R-text)


Contents: ####

1. Start here: https://keras.rstudio.com/index.html (which is the same as  https://tensorflow.rstudio.com/keras)
Then, as instructed there go to. # Learning More:

1-2. Guide to the Sequential Model - https://keras.rstudio.com/articles/sequential_model.html
Then, as instructed there go to # Examples:

[1-2-1]. CIFAR10 small images classification -      https://keras.rstudio.com/articles/examples/cifar10_cnn.html
1-2-2. IMDB movie review sentiment classification - https://keras.rstudio.com/articles/examples/imdb_cnn_lstm.html
1-2-3. Reuters newswires topic classification -     https://keras.rstudio.com/articles/examples/reuters_mlp.html
1-2-4. MNIST handwritten digits classification -    https://keras.rstudio.com/articles/examples/mnist_mlp.html


1-3. Guide to the Functional API -    https://keras.rstudio.com/articles/functional_api.html
[1-4]. Frequently Asked Questions -   https://keras.rstudio.com/articles/faq.html
1-1. Training Visualization -         https://keras.rstudio.com/articles/training_visualization.html

Other files:
- DLwR-s6.1-RNN-for-text.R
- DLwR-s6.1-RNN-forSequences.R
- DLwR-s3-IMDB_sentimentBinary+wiresClassification+housepriceReression.R


Libraries used (and highly recommended for efficient programming):   
`library(ggplot2);library(data.table); library(magrittr); library(lubridate); library(stringr)`
