#
# https://tensorflow.rstudio.com/tensorflow/articles/using_tensorflow_api.html
#

library(tensorflow)

sess = tf$Session()

hello <- tf$constant('Hello, TensorFlow!')
sess$run(hello)

a <- tf$constant(10L)
b <- tf$constant(32L)
sess$run(a + b)






# Python API Reference: https://www.tensorflow.org/api_docs/python/

# Python 	        R 	Examples
# Scalar 	        Single-element vector 	    1, 1L, TRUE, "foo"
# List 	          Multi-element vector 	      c(1.0, 2.0, 3.0), c(1L, 2L, 3L)
# Tuple 	        List of multiple types 	    list(1L, TRUE, "foo")
# Dict 	          Named list or dict 	        list(a = 1L, b = 2.0), dict(x = x_data)
# NumPy ndarray 	Matrix/Array 	              matrix(c(1,2,3,4), nrow = 2, ncol = 2)
# None, True, False 	NULL, TRUE, FALSE 	    NULL, TRUE, FALSE


# NumPy Types

np <- import("numpy")
training_set <- tf$contrib$learn$datasets$base$load_csv_with_header(
  filename = "iris_training.csv",
  target_dtype = np$int,
  features_dtype = np$float32
)
