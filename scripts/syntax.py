# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()    # Because I am using tensorflow v1
    

######################################################################
# BASIC COMPUTATIONS
######################################################################


hello = tf.constant("Hello")

world = tf.constant(" world!")

with tf.Session() as sess:
    result = sess.run(hello + world)

a = tf.constant(10)

b = tf.constant(20)

with tf.Session() as sess:
    addition_output = sess.run(a + b)


######################################################################
# OPERATIONS
######################################################################

const = tf.constant(10)

fill_mat = tf.fill((4, 4), 10)    # Fill matrix, basically an array with dimensions - 4 x 4 with 10 as the value.

zeros = tf.zeros((4, 4)) # Creates a tensor with all elements set to zero. 

ones = tf.ones((4, 4)) # Creates a tensor with all elements set to one. 

randn = tf.random_normal((4, 4), mean=0, stddev=1.0) # Random normal distribution. Mean, standard deviation.

randu = tf.random_uniform((4, 4), minval=0, maxval=1) # A uniform distribution - min value and max value. 

ops = [const, fill_mat, zeros, ones, randn, randu]

with tf.Session() as sess:
    for op in ops:
        result = sess.run(op)
        # print(result)

"""
Output of above: 

-----------------------------------------------------
const: 

10
-----------------------------------------------------
fill_mat:

[[10 10 10 10]
 [10 10 10 10]
 [10 10 10 10]
 [10 10 10 10]]
-----------------------------------------------------
zeros: 

[[0. 0. 0. 0.]
 [0. 0. 0. 0.]
 [0. 0. 0. 0.]
 [0. 0. 0. 0.]]
 -----------------------------------------------------
 ones:

[[1. 1. 1. 1.]
 [1. 1. 1. 1.]
 [1. 1. 1. 1.]
 [1. 1. 1. 1.]]
 -----------------------------------------------------
 randn: 

[[-0.56681347  0.30550814  0.07210699  0.74969715]
 [-1.7469424  -0.8448174  -0.6570759  -2.7031488 ]
 [-2.86716    -0.62029415 -2.3770144  -1.0596708 ]
 [ 1.5663337  -1.0772523  -1.829789   -0.9909293 ]]
 -----------------------------------------------------
 randu:

[[0.30300665 0.73948526 0.46500254 0.7926736 ]
 [0.00840807 0.92443776 0.83706427 0.3222834 ]
 [0.27990222 0.4652288  0.8173822  0.433326  ]
 [0.20070374 0.03991103 0.86316264 0.98646903]]
 -----------------------------------------------------

"""


######################################################################
# MATRIX MULTIPLICATION
######################################################################


a = tf.constant([ [1, 2],
                  [3, 4] ]) # Nested list.

a.get_shape()
# print(a.get_shape())

b = tf.constant([ [10], [100] ])

b.get_shape()
# print(b.get_shape())

result = tf.matmul(a, b) # Multiply matrices

with tf.Session() as sess:
    sess.run(result)
    # print(sess.run(result))

    