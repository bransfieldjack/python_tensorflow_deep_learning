# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()    # Because I am using tensorflow v1
import numpy as np

"""

This neuron performs a linear fit to some two dimensional data. 

Steps:

1). Build the graph.
2). Initiate a session.
3). Feed data + output

Graph: 

 wx + b = z

"""

######################################################################
# MODEL
######################################################################

np.random.seed(101) # Set random seed values 
tf.set_random_seed(101) # Set random seed values 

rand_a = np.random.uniform(0, 100, (5, 5))  # Variable
rand_b = np.random.uniform(0, 100, (5, 1))  # Variable

a = tf.placeholder(tf.float32) # Placeholder
b = tf.placeholder(tf.float32) # Placeholder

"""
Add some operations, additive and multiplicative:
"""

add_op = a + b
mul_op = a * b

"""
Start a session and input data to the operation. Data must be input using a 'feed_dict' with TF:
"""

with tf.Session() as sess:
    add_result = sess.run(add_op, feed_dict={a: 10, b: 20})
    # print(add_result)

"""
Confirm that the above is working, then supplement data:
"""

with tf.Session() as sess:
    add_result = sess.run(add_op, feed_dict={a: rand_a, b: rand_b})
    mult_result = sess.run(mul_op, feed_dict={a: rand_a, b: rand_b})
    # print(mult_result)


