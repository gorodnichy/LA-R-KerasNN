# RStudio-tf-HelloMNIST.R
# Sources: ----
# https://github.com/rstudio/tensorflow/tree/master/inst/examples -> ./rstudio_tensorflow_examples
# https://tensorflow.rstudio.com/tensorflow/
# https://tensorflow.rstudio.com/tensorflow/articles/basic_usage.html
# https://tensorflow.rstudio.com/tensorflow/articles/tutorial_mnist_beginners.html
# https://tensorflow.rstudio.com/tensorflow/articles/tutorial_mnist_pros.html
# https://tensorflow.rstudio.com/tensorflow/articles/tutorial_tensorflow_mechanics.html
    # Source: https://github.com/rstudio/tensorflow/blob/master/inst/examples/mnist/fully_connected_feed.R


library(tensorflow)

# . 1 Hello ----

sess = tf$Session()

hello <- tf$constant('Hello, TensorFlow!')
sess$run(hello)

a <- tf$constant(10L)
b <- tf$constant(32L)
sess$run(a + b)


# . 2 Launching the graph in a session ----


# Create a Constant op that produces a 1x2 matrix.  The op is
# added as a node to the default graph.
#
# The value returned by the constructor represents the output
# of the Constant op.
matrix1 <- tf$constant(matrix(c(3.0, 3.0), nrow = 1, ncol = 2))

# Create another Constant that produces a 2x1 matrix.
matrix2 <- tf$constant(matrix(c(3.0, 3.0), nrow = 2, ncol = 1))

# Create a Matmul op that takes 'matrix1' and 'matrix2' as inputs.
# The returned value, 'product', represents the result of the matrix
# multiplication.
product <- tf$matmul(matrix1, matrix2)


# https://www.tensorflow.org/api_guides/python/client#session-management
sess <- tf$Session()

# To run the matmul op we call the session 'run()' method, passing 'product'
# which represents the output of the matmul op.  This indicates to the call
# that we want to get the output of the matmul op back.
#
# All inputs needed by the op are run automatically by the session.  They
# typically are run in parallel.
#
# The call 'run(product)' thus causes the execution of three ops in the
# graph: the two constants and matmul.
#
# The output of the op is returned in 'result' as a 1x1 matrix.
result <- sess$run(product)
print(result)

# Close the Session when we're done.
sess$close()

# . 2b Interactive Usage ----

# Enter an interactive TensorFlow Session.

sess <- tf$InteractiveSession()

x <- tf$Variable(c(1.0, 2.0))
a <- tf$constant(c(3.0, 3.0))

# Initialize 'x' using the run() method of its initializer op.
x$initializer$run()

# Add an op to subtract 'a' from 'x'.  Run it and print the result
sub <- tf$subtract(x, a)
print(sub$eval())

# Close the Session when we're done.
sess$close()

# .2c Variables ----

# Create a Variable, that will be initialized to the scalar value 0.
state <- tf$Variable(0L, name="counter")

# Create an Op to add one to `state`.
one <- tf$constant(1L)
new_value <- tf$add(state, one)
update <- tf$assign(state, new_value)

# Variables must be initialized by running an `init` Op after having
# launched the graph.  We first have to add the `init` Op to the graph.
init_op <- tf$global_variables_initializer()

# Launch the graph and run the ops.
with(tf$Session() %as% sess, {
  # Run the 'init' op
  sess$run(init_op)
  # Print the initial value of 'state'
  print(sess$run(state))
  # Run the op that updates 'state' and print 'state'.
  for (i in 1:3) {
    sess$run(update)
    print(sess$run(state))
  }
})

#sess$close()

# .2d Fetches and Feeds ----

input1 <- tf$constant(3.0)
input2 <- tf$constant(2.0)
input3 <- tf$constant(5.0)
intermed <- tf$add(input2, input3)
mul <- tf$multiply(input1, intermed)

with(tf$Session() %as% sess, {
  result = sess$run(list(mul, intermed))
  print(result)
})


input1 <- tf$placeholder(tf$float32)
input2 <- tf$placeholder(tf$float32)
output <- tf$multiply(input1, input2)

with(tf$Session() %as% sess, {
  print(sess$run(output, feed_dict=dict(input1 = 7.0, input2 = 2.0)))
})


# . 3 MNIST ----


#https://raw.githubusercontent.com/rstudio/tensorflow/master/inst/examples/mnist/fully_connected_feed.R
source("fully_connected_feed.R", echo = TRUE)

source("https://raw.githubusercontent.com/rstudio/tensorflow/master/inst/examples/mnist/mnist.R")



source("https://raw.githubusercontent.com/rstudio/tensorflow/master/inst/examples/mnist/fully_connected_feed.R", echo = TRUE)


