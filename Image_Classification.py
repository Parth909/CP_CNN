import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# ====> 1. LOOKING & LOADING THE DATA

data = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = data.load_data()

# The labels are an array of integers, ranging from 0 to 9.
# These correspond to the class of clothing the image represents:
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# train_images is a collection of 60,000 images
# An 3D containing 60,000 2D Arrays
# Shape of 2D array is 28 X 28 (i.e :- 28 1D Arrays with 28 Items in it)
# Each item can have value between 0 to 255
print(train_images.shape)
print(train_images[7])

# Showing the Image. We can see the image bcz it is 2D array of shape 28 X 28
# plt.imshow(train_images[7])
# plt.show()

# Converting those values to be between 0 & 1
train_images = train_images/255.0 # Can directly divide bcz it is a NUMPY array
test_images = test_images/255.0
print("IMAGE DATA \n",train_images[7])

# ====> 2. CREATING THE MODEL

# NOTE :- We cannot directly pass in the 2D Data to our Neural Network we need to flatten it out
# Convert 28 x 28 Data to a 1D Array of length 784

# keras.Sequential :- Means Sequence of Layers
model = keras.Sequential([
    # Input Layer
    keras.layers.Flatten(input_shape=(28,28)),
    # Hidden Layer, Activation Function :- Rectify Linear Unit
    keras.layers.Dense(128, activation="relu"),
    # Output Layer, Activation Function :- "softmax" means like 80 percent this 20 percent that like that
    keras.layers.Dense(10, activation="softmax")
    # In any neural network, a dense layer is a layer that is deeply connected with its preceding layer
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# train the model
print(len(train_images))
model.fit(train_images, train_labels, epochs=5)

# testing the model
test_loss, test_acc = model.evaluate(test_images, test_labels)

print("Tested Acc :- ", test_acc)

# ====> 3. USING & TESTING THE MODEL

# put our 28 x 28 Image Array inside a LIST bcz model.predict takes data inside the LIST

# NOTE :- If we use this it gives the following error :-
# prediction = model.predict([test_images[7]])
# tensorflow:Model was constructed with shape (None, 28, 28)

# SOLUTION :-

img = np.expand_dims(test_images[7], axis=0)

# print("======= img using [img] ===== \n", [test_images[7]])
# print("======= img using np.array(test_images[7]) ===== \n", np.array(test_images[7]))
# print("======= img using np.expand_dims(test_images[7], axis=0) =====\n", np.expand_dims(test_images[7], axis=0))

# From above only np.expand_dims() gives the correct data type which is "[[[(0,0,0,0,0,0,0.72,0.22), ......]]]"

# Others give this data type [array([[(0,0,0,0,0,0,0.72,0.22), ......]])] which doesn't work properly when predicting

prediction = model.predict(img)

print("The prediction of a single image ", prediction)

# to predict all of them
prediction2 = model.predict(test_images)

print("The prediction of all the images\n", prediction2)

# for every Image (2D Array) there will be 10 predictions
print("The prediction of the first image is\n", prediction2[0])

i = np.argmax(prediction2[0])

print("The highest prediction, Index of highest prediction\n", i)

print("The fashion item belongs to the class", class_names[i])

# See the actual image to confirm if the Prediction is correct or not
plt.imshow(test_images[0])
plt.show()