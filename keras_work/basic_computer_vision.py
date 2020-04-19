import tensorflow as tf
from tensorflow import keras


"""

This is a basic script for teaching a machine how to recognise different objects. 
If you take the example of a shoe, you can recognise visually that a high heel and a rugby boot are both different types of shoes. 
But why do you know this? How would someone who has never seen a shoe in their life make the same distinction? 
You know this because during your lifetime, you have seen lots of different shoes and you have learned to distinguish between 
all the different variations. 
The same methodology can be applied to machine learning using computer vision.

There is a popular sample dataset for training purposes known as 'Fashion MNIST'. 
This dataset has 70K images in 10 different categories. 
So there are 7K examples of each category, including shoes. 
The hope is that seeing 7K different types of shoe, will be enough for a computer to familiarise itself with what a shoe looks like.

The images contained in 'Fashion MNIST' are only 28x28 pixels - pretty small.
The less data used, the faster it is for a machine to process it - Fashion MNIST is well suited then.
That being said, the images still need to contain recognisable items of clothing, 
you should basically be able to roughly make out what you are looking at.

Tensorflow really is a great tool because it allows you to utilise a consitent API to design different neural networks that 
are suited to different learning tasks. 

The following is the code required to achieve everything I've just written about above:

"""


fashion_mnist = keras.datasets.fashion_mnist    # the fashion mnist dataset is built into tensorflow. 


"""
The training images, is a set of 60K images.
The remaining 10K images are a test set to check how well the neural net is performing. 
The label is a number indicating the class of that type of clothing, for example: 09 - indicates an ankle boot etc. 
There are two main reasons for why the label is represented by a number and not just the text 'ankle boot': 
(1) machines deal better with numbers, (2) bias: if we label it as ankle boot, we're already showing a bias towards
the english language.
"""


(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()


"""
When looking at a neural net design, its a good idea to first look at the input and output values. 

(first layer:)
The first layer of the neural net for this CV model has the shape 28 x 28.
This is actually the size of each image (28x28 pixels).

(second layer:)
There are 128 functions. The combination of all of these functions should output the correct value. 
In order to do that, the machine needs to figure out the parameters inside each function, to return the desired result. 
It will then extend this to all of the other items of clothing in the dataset. 
Once it has done this, it should be able to recognise items of clothing. 

(last layer:)
The last layer is 10, this is the number of different items of clothing represented in the dataset.
The neural net is kind of acting like a filter, taking in a 28x28 set of pixels and outputs one of ten values. 
"""

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])


"""
The neural net will be initialised with random values. 
The loss function will measure how good/bad the results were, and the optimizer will generate new parameters 
for the functions to improve its performance. 

The activations functions: 

    The second layer of 128 functions, this is relu (rectified linear unit).
    RELU returns a value if its greater than 0.
    If the function returns 0 or less as output, it is discarded. 

    Softmax - this picks the biggest number in a set. 
    Is basically sets the returned correct value to one, and all other values (in 10) to 0.

"""

model.compile(optimizer=tf.train.AdamOptimizer(),
              metrics=['accuracy'],
              loss='sparse_categorical_crossentropy')

        
"""
Training then becomes simple. 
Fir the training images to the training labels. 
Below is an example of this for 5 epochs:
"""

model.fit(train_images, train_labels, epochs=5)


"""
The test data - remember I still have 10K images that the model doesnt know about in reserve. 
We can use these to test how well the model performs. 
We can do that test by passing them to the evaluate method: 
"""

test_loss, test_acc = model.evaluate(test_images, test_labels)

"""
Finally we can get predictions back for new images by calling model.predict:
"""

# model.predict(my_images)
model.evaluate(test_images, test_labels)





