import keras
import numpy as np

"""
This is an example of a machine learning model used to match one set of numbers to another, for example x, y using y = mx - c etc.
For example: 

X = -1, 0, 1, 2, 3, 4
Y = -3, -1, 1, 3, 5, 7


Visually, you can conclude that there is a relationship between these two axis. (just picture the points on the graph in your head)
Y = 2X - 1

This is machine learning in essence, you provide the answers and the input and the machine learns the relationship.

The following is code that creates a machine learned model to figure out what matches these numbers to each other:
"""


"""
The first line defines the model itself.
A model is a trainned neural network, here we have the simplest possible neural network. 
In this case, the model has a single layer - indicated by the 'keras.layers.Dense' piece of code.
The layer has a single neuron in it, indicated by 'units=1'.
We also feed a single value into the neural network (x value) the neural network will predict what the y will be for that x.
Thats why we say that the input shape is just one value. 
"""

model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])


"""
When you compile the model, there are two functions - loss and optimizer. 
The model will make a guess about the relationship between the numbers, for example y = 5x + 5...
When training, it will calculate how good or how bad the guess is, using the loss function. 
Then, it will use the optimizer function to generate another guess. 
The logic being that the combination of these two functions will get us closer to the correct formula. 
"""

model.compile(optimizer='sgd', loss='mean_squared_error')


xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)


"""
In this model, it will go through the training loop 500 times (epoch).
It will make a guess, calculate how accurate the guess is, then use the optimixer to enhance that guess.
The process of matching the above data is in the 'fit' method of the model.
"""

model.fit(xs, ys, epochs=500)

print(model.predict([10.0]))

"""
The result of the above print operation will look like this: 

Epoch 489/500
6/6 [==============================] - 0s 115us/step - loss: 9.0541e-05
Epoch 490/500
6/6 [==============================] - 0s 277us/step - loss: 8.8681e-05
Epoch 491/500
6/6 [==============================] - 0s 487us/step - loss: 8.6860e-05
Epoch 492/500
6/6 [==============================] - 0s 113us/step - loss: 8.5076e-05
Epoch 493/500
6/6 [==============================] - 0s 92us/step - loss: 8.3328e-05
Epoch 494/500
6/6 [==============================] - 0s 91us/step - loss: 8.1618e-05
Epoch 495/500
6/6 [==============================] - 0s 90us/step - loss: 7.9941e-05
Epoch 496/500
6/6 [==============================] - 0s 92us/step - loss: 7.8298e-05
Epoch 497/500
6/6 [==============================] - 0s 91us/step - loss: 7.6690e-05
Epoch 498/500
6/6 [==============================] - 0s 101us/step - loss: 7.5116e-05
Epoch 499/500
6/6 [==============================] - 0s 97us/step - loss: 7.3572e-05
Epoch 500/500
6/6 [==============================] - 0s 99us/step - loss: 7.2062e-05
[[18.975233]]

^ The returned value is 18.975233, this is because the model was trained with 6 values, 
but we asked it to make a prediction for 10 values, 
so it did this to the best possible value which was float 18.975233.

"""