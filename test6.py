# Importing libraries
import numpy as np
import pandas as pd
import tensorflow as tf
import os
import time
import hashlib
import requests
import gzip

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



pset_5_cache = ""
X_train = fetch_np_array_new("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz", pset_5_cache)[0x10:].reshape((-1, 28, 28))
y_train = fetch_np_array("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz", pset_5_cache)[8:]
X_test = fetch_np_array("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz", pset_5_cache)[0x10:].reshape((-1, 28, 28))
y_test = fetch_np_array("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz", pset_5_cache)[8:]


# Now we need to divide them all by 255
X_train = X_train/255.0
X_test = X_test/255.0

# Next we need to reshape our data for the convolutional network
X_train = X_train.reshape(X_train.shape[0], 28, 28,1)
X_test = X_test.reshape(X_test.shape[0], 28, 28,1)



# Creating our MNIST network
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3),activation='relu',kernel_initializer='he_normal',input_shape=(28,28,1)))
model.add(tf.keras.layers.MaxPool2D((2, 2)))
model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu',kernel_initializer='he_normal'))
model.add(tf.keras.layers.MaxPool2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu',kernel_initializer='he_normal'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dense(10, activation='softmax'))

# Compiling model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Training model
model.fit(X_train, y_train, epochs=5)

# Creating our predictions for submission
preds = pd.DataFrame({'ImageId': list(range(1,test.shape[0]+1)),'Label': model.predict_classes(X_test)})
preds.to_csv('submission.csv', index=False, header=True)

