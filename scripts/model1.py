# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()    # Because I am using tensorflow v1
import numpy as np
import matplotlib
matplotlib.use('WebAgg')    # Using this backend to render my matplotlib plots. Serving on: http://127.0.0.1:8988/
import matplotlib.pyplot as plt


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


"""
An example eneural network (graph wx + b = z) : 
"""

n_features = 10
n_dense_neurons = 3

x = tf.placeholder(tf.float32, (None, n_features))  # Placeholder x

W = tf.Variable(tf.random_normal([n_features, n_dense_neurons])) # Variable W

b = tf.Variable(tf.ones([n_dense_neurons]))

xW = tf.matmul(x, W)

z = tf.add(xW, b)

a = tf.sigmoid(z) # Activation function 

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    layer_out = sess.run(a, feed_dict={x: np.random.random([1, n_features])})
    # print(layer_out)

"""
The above runs a neural net taking in random value variables w & b.
The following is a pre determined regression example:
"""

x_data = np.linspace(0, 10, 10) + np.random.uniform(-1.5, 1.5, 10)

y_label = np.linspace(0, 10, 10) + np.random.uniform(-1.5, 1.5, 10)

plt.plot(x_data, y_label, 'go') # 'Go' plots with points instead of lines. 

# plt.show()

"""
Make a neural network.
y = mx + c
"""

m = tf.Variable(0.44)   # Init with random values 
b = tf.Variable(0.87)   # Init with random values 

"""
Create the cost function. 
'measures the performance of a Machine Learning model for given data'
"""

error = 0

for x, y in zip(x_data, y_label):   # list of tuples

    y_hat = m * x + b   # predicted value, this will be off becuase im using random value. Fix this with cost function. 
    error += (y-y_hat)**2   # y is the real value, subtract the predicted value. You want to minimise this error to produce an effective cost function. 
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001) # In order to minimise you need to use some sort of optimizer. 
    train = optimizer.minimize(error)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        
        sess.run(init)
        training_steps = 1000 # specify how many training steps being performed. 

        for i in range(training_steps):
            sess.run(train)

        final_slope, final_intercept = sess.run([m, b])

x_test = np.linspace(-1, 11, 10)

y_pred_plot = final_slope * x_test + final_intercept

plt.plot(x_test, y_pred_plot, 'r')
plt.plot(x_data, y_label, 'go')

plt.show()

