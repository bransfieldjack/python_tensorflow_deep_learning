import tensorflow as tf
from tensorflow import keras

"""

https://www.youtube.com/watch?v=x_VrgWTKkiM

Convolutional neural networks are useful for distinguishing between shapes contained in a single image. 
Basic computer vision using a deep neural network is appropriate for a dataset of images such as fashion MNIST 28x28 pixels.
This is a limitation, the images in that dataset all have their subjects centered and it was the only thing in the image. 

The idea behind a CNN, is that you filter the images before training the deep neural network. 
After filtering the images, features within them can then come to the forefront, and you can distinguish identifiable features. 

A filter is simply a set of multiplyers. 

"""

model = tf.keras.models.Sequential([  
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu',   # Generates 64 filters, multiplying each of them acorss the image.
                           input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),  # Dont specify an input shape, instead put a convolutional layer on top of it like above ^
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

