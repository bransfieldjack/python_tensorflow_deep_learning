# https://www.tensorflow.org/tutorials/keras/text_classification_with_hub

"""
** Make sure your running tf version 2.1.0


Classifies movie reviews as positive or negative using the text of the review.
This is a binary classification problem, it will classify a review based on a positive/negative piece of text. 
This is an investigative excercise. Ideally I would like to expand on this to use a multi-class classification model to 
assist with classification of diverse stem cell datasets based on the api.stemformatics.org dataportal repository. 
For example, a binary classification model could possibliy be applied to the atlas project to classify blood/iMac datasets. 

*
Transfer learning is a research problem in machine learning that focuses on storing knowledge gained while 
solving one problem and applying it to a different but related problem. For example, 
knowledge gained while learning to recognize cars could apply when trying to recognize trucks.
*

I'm using the IMDB movie datasets for this. 50k in total, 25k training and 25k test. All sets are balanced.

"""

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds


print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("Hub version: ", hub.__version__)
print("GPU is", "available" if tf.config.experimental.list_physical_devices("GPU") else "NOT AVAILABLE")

"""
Downloading the IMDB dataset:
"""

# Split the training set into 60% and 40%, so we'll end up with 15,000 examples
# for training, 10,000 examples for validation and 25,000 examples for testing.
train_data, validation_data, test_data = tfds.load(
    name="imdb_reviews", 
    split=('train', 'train', 'test'),
    as_supervised=True) # If true, returns dataset as 2 x tuple structure. If false, returns dict. 


"""
Taking a look at the data:
"""

train_examples_batch, train_labels_batch = next(iter(train_data.batch(10)))
#print(train_examples_batch)
#print(train_labels_batch)


"""
As you now know, neural networks are defined by their 'layers'.
There are some design decisions you should consider when building the model for this... 

1: How to represent the text?
2: How many layers should be used/are required?
3: How many so called 'hidden units' to use in each layer. 

In this example, the input data consists of sentences. The labels to predict are either 0 or 1, true/false.

"

One way to represent the text is to convert sentences into embeddings vectors. 
We can use a pre-trained text embedding as the first layer, which will have three advantages:

- we don't have to worry about text preprocessing,
- we can benefit from transfer learning,
- the embedding has a fixed size, so it's simpler to process.
 
For this example we will use a pre-trained text embedding model from TensorFlow Hub called google/tf2-preview/gnews-swivel-20dim/1.

"

* Notes: 

    https://www.quora.com/What-is-Continuous-Vector-Representations-of-words
"""

embedding = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"
hub_layer = hub.KerasLayer(embedding, input_shape=[], 
                           dtype=tf.string, trainable=True)
#print(hub_layer(train_examples_batch[:3]))
hub_layer(train_examples_batch[:3])

"""
Build the model: 
"""

model = tf.keras.Sequential()
model.add(hub_layer)
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(1))

print(model.summary())

"""
Configure the optimizer and loss functions: 
"""

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

"""
Start the training: 
"""

history = model.fit(train_data.shuffle(10000).batch(512),
                    epochs=20,
                    validation_data=validation_data.batch(512),
                    verbose=1)

"""
Evalutation: 
"""

results = model.evaluate(test_data.batch(512), verbose=2)

for name, value in zip(model.metrics_names, results):
  print("%s: %.3f" % (name, value))


