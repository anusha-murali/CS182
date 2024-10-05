# -*- coding: utf-8 -*-
"""
CS 182 Pset4: Python Coding Questions - Fall 2023
Due December 4, 2023 at 11:59pm
"""


#### Coding Problem Set General Instructions - PLEASE READ ####
# 1. All code should be written in python 3.7 or higher to be compatible with the autograder
# 2. Your submission file must be named "pset4b.py" exactly
# 3. No additional outside packages can be referenced or called, they will result in an import error on the autograder
# 4. Function/method/class/attribute names should not be changed from the default starter code provided
# 5. All helper functions and other supporting code should be wholly contained in the default starter code declarations provided.
#    Functions and objects from your submission are imported in the autograder by name, unexpected functions will not be included in the import sequence

###############################
# Question 5: Neural Networks #
###############################

### Package Imports ###
import requests, gzip, os, hashlib
import tensorflow as tf
import abc
import numpy as np
import time
### Package Imports ###

# Instructions: Fill in the `train` and `predict` functions for the tensorflow model, which inherits from 
# `MNISTClassificationModel`. See that class's docstring for details. You should use the `neural_net_model` attribute 
# to hold the actual neural network. In particular, make sure the assertions in each `__init__` method passes (DO NOT 
# DELETE THOSE ASSERTIONS!).

# The autograder will give up to 3 minutes to train and evaluate your model. We will check that after training and
# predicting according to the procedure that you code up, your models have an accuracy of at least 0.95.

# To learn how to use each TensorFlow, you may want to reference the official documentation or an online tutorial
# This assignment is largely an exercise in learning how to use third-party machine learning libraries.

def fetch_np_array(url: str, path: str) -> np.ndarray:
    """
    This function retrieves the data from the given url and caches the data locally in the path so that we do not
    need to repeatedly download the data every time we call this function.

    Args:
        url: link from which to retrieve data
        path: path on local desktop to save file

    Returns:
        Numpy array that is fetched from the given url or retrieved from the cache.
    """
    fp = os.path.join(path, hashlib.md5(url.encode('utf-8')).hexdigest())
    if os.path.isfile(fp):
        with open(fp, "rb") as f:
            data = f.read()
    else:
        with open(fp, "wb") as f:
            data = requests.get(url).content
            f.write(data)
    return np.frombuffer(gzip.decompress(data), dtype=np.uint8).copy()


def fetch_np_array_new(url: str, path: str) -> np.ndarray:
    """
    This function retrieves the data from the given url and caches the data locally in the path so that we do not
    need to repeatedly download the data every time we call this function.

    Args:
        url: link from which to retrieve data
        path: path on local desktop to save file

    Returns:
        Numpy array that is fetched from the given url or retrieved from the cache.
    """
    fp = os.path.join("", hashlib.md5(url.encode('utf-8')).hexdigest())
    with open(fp, "wb") as f:
            data = requests.get(url).content
            print(np.shape(data))
            f.write(data)
    return np.frombuffer(gzip.decompress(data), dtype=np.uint8).copy()



class MNISTClassificationModel(abc.ABC):
    neural_net_model = None

    @abc.abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Args:
            X_train: Numpy array of shape (num_samples, 28, 28) of integer values, from 0...255 for color intensity.
            y_train: Numpy array of shape (num_samples,), containing integer labels
        """
    @abc.abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Args:
            X: Numpy array of shape (num_samples, 28, 28) of integer values, from 0...255 for color intensity.

        Returns:
            y: Numpy array of shape (num_samples,) of integer-valued labels.
        """

class TensorFlowMNISTClassifier(MNISTClassificationModel):
    def __init__(self):
        """
        Initialize your `neural_net_model`.
        """
        ##########################
        ##### YOUR CODE HERE #####
        ##########################
        self.neural_net_model = tf.keras.models.Sequential([\
            tf.keras.layers.Flatten(input_shape=(28, 28)),\
##            tf.keras.layers.Dense(128, activation='relu'),\
##            tf.keras.layers.Dropout(0.2),\
##            tf.keras.layers.Dense(10)])
            tf.keras.layers.Dense(5, activation='sigmoid'),\
            tf.keras.layers.Dense(10, activation='softmax')])
        
        # self.neural_net_model = tf.Module ()

##        self.neural_net_model.compile(optimizer='sgd',
##              loss='mse',
##              metrics=['accuracy'])

        self.neural_net_model.compile(loss='categorical_crossentropy', 
              optimizer='adam',
              metrics=['acc'])
        
        assert isinstance(self.neural_net_model, tf.Module)

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Fitting proceadure of the network on an inut X_train and y_train data set"""
        ##########################
        ##### YOUR CODE HERE #####
        ##########################
        self.neural_net_model.compile(optimizer='sgd',
              loss='mse',
              metrics=['accuracy'])

        # self.neural_net_model.fit(X_train, y_train, epochs=1)
        

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Network forward pass, for an input dataset X, return the output predictions"""
        ##########################
        ##### YOUR CODE HERE #####
        ##########################
        raise NotImplementedError


def train_and_evaluate_mnist_model(
        model: MNISTClassificationModel,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray
        ) -> float:
    """
    Evaluates the model provided to this function given the appropriate inputs.

    Returns:
        Prediction accuracy on test set
    """
    start = time.time()
    model.train(X_train, y_train)
    end = time.time()
    print(f"Training time is {end - start} seconds.")
    y_pred = model.predict(X_test)
    number_correct = (y_pred == y_test).sum()
    return number_correct / len(y_test)

def display():
    import matplotlib.pyplot as plt
    
    # Display some images
    fig, axes = plt.subplots(ncols=5, sharex=False, sharey=True, figsize=(10, 4))
    for i in range(5):
        axes[i].set_title(y_train[i])
        axes[i].imshow(X_train[i], cmap='gray')
        axes[i].get_xaxis().set_visible(False)
        axes[i].get_yaxis().set_visible(False)
    plt.show()

if __name__ == "__main__":
    # Obtain data from the web
    pset_5_cache = ""
    X_train = fetch_np_array_new("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz", pset_5_cache)[0x10:].reshape((-1, 28, 28))
    y_train = fetch_np_array("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz", pset_5_cache)[8:]
    X_test = fetch_np_array("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz", pset_5_cache)[0x10:].reshape((-1, 28, 28))
    y_test = fetch_np_array("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz", pset_5_cache)[8:]

    X_train = X_train/255.0
    X_test = X_test/255.0

    from keras.utils import to_categorical
    
    # Convert y_train into one-hot format
    temp = []
    for i in range(len(y_train)):
        temp.append(to_categorical(y_train[i], num_classes=10))
    y_train = np.array(temp)
    # Convert y_test into one-hot format
    temp = []
    for i in range(len(y_test)):    
        temp.append(to_categorical(y_test[i], num_classes=10))
    y_test = np.array(temp)
    
    # Initialize tensorflow model for training and evaluation
##    tf_model = TensorFlowMNISTClassifier()
##    tf_accuracy = train_and_evaluate_mnist_model(tf_model, X_train, y_train, X_test, y_test)
##    print(f"Accuracy of tf model is {tf_accuracy}.")

    from keras.layers import Dense, Flatten
    from keras.models import Sequential
    
    model = Sequential()
    model.add(Flatten(input_shape=(28,28)))
    model.add(Dense(5, activation='sigmoid'))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy', 
	      optimizer='adam',
	      metrics=['accuracy'])

    model.summary()

##    # Train the Neural Network model
##    model.fit(X_test, y_test, epochs=1, validation_data=(X_test,y_test))
##
##    # Making predictions using our trained model
##    predictions = model.predict(X_test)
##    predictions = np.argmax(predictions, axis=1)
##
##    # Display some predictions on test data
##    fig, axes = plt.subplots(ncols=10, sharex=False,
##			 sharey=True, figsize=(20, 4))
##    for i in range(10):
##        axes[i].set_title(predictions[i])
##        axes[i].imshow(X_test[i], cmap='gray')
##        axes[i].get_xaxis().set_visible(False)
##        axes[i].get_yaxis().set_visible(False)
##    plt.show()

    

