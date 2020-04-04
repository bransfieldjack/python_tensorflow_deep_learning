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
# BASIC COMPUTATIONS
######################################################################