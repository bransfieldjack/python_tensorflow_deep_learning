# https://www.tensorflow.org/tutorials/keras/text_classification_with_hub

"""

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
train_examples_batch