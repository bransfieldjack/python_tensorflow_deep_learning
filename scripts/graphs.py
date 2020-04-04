# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()    # Because I am using tensorflow v1

"""
TF makes use of graphs behind the scenes.
Graphs are sets of connected nodes, called vertices. 
Connections between nodes are called edges. 
Each node, is an operation that takes input(s) and produces an output. 

This script will construct a graph and execute it. 
"""

######################################################################
# GRAPHS
######################################################################

n1 = tf.constant(1)

n2 = tf.constant(2)

n3 = n1 + n2

with tf.Session() as sess:
    result = sess.run(n3)
    # print(result)

# Tensorflow creates a default graph when its started. 
# You can check it with print(tf.get_default_graph())

g = tf.Graph()  # This adds an additional graph to tensorflow. 

graph_one = tf.get_default_graph()  # The current default graph. 

graph_two = tf.Graph()  # Creates a new tensorflow graph. 

"""
If I wanted to assign graph_two as the default graph withing the session:
"""

with graph_two.as_default():
    print(graph_two is tf.get_default_graph())


"""
Trying to assign this outside of the session will result in the default graph being printed as true (graph_one).
"""




