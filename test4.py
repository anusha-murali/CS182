import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense, Flatten
from keras.models import Sequential
from keras.utils import to_categorical
from keras.datasets import mnist
import ssl

# Load MNIST handwritten digit data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Display some images
fig, axes = plt.subplots(ncols=5, sharex=False,
			 sharey=True, figsize=(10, 4))
for i in range(5):
	axes[i].set_title(y_train[i])
	axes[i].imshow(X_train[i], cmap='gray')
	axes[i].get_xaxis().set_visible(False)
	axes[i].get_yaxis().set_visible(False)
plt.show()

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

# Create simple Neural Network model
model = Sequential()
model.add(Flatten(input_shape=(28,28)))
model.add(Dense(5, activation='sigmoid'))
model.add(Dense(10, activation='softmax'))

model.summary()

##model.compile(loss='categorical_crossentropy', 
##	      optimizer='adam',
##	      metrics=['accuracy'])

### Train the Neural Network model
##model.fit(X_train, y_train, epochs=1, validation_data=(X_test,y_test))
##
### Making predictions using our trained model
##predictions = model.predict(X_test)
##predictions = np.argmax(predictions, axis=1)
##
### Display some predictions on test data
##fig, axes = plt.subplots(ncols=10, sharex=False,
##			 sharey=True, figsize=(20, 4))
##for i in range(10):
##	axes[i].set_title(predictions[i])
##	axes[i].imshow(X_test[i], cmap='gray')
##	axes[i].get_xaxis().set_visible(False)
##	axes[i].get_yaxis().set_visible(False)
##plt.show()
