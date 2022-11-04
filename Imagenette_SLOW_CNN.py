# Fetch data :-

imagenette_map = {
    "n01440764" : "tench",
    "n02102040" : "springer",
    "n02979186" : "casette_player",
    "n03000684" : "chain_saw",
    "n03028079" : "church",
    "n03394916" : "French_horn",
    "n03417042" : "garbage_truck",
    "n03425413" : "gas_pump",
    "n03445777" : "golf_ball",
    "n03888257" : "parachute"
}

# 1. Loading images without maxing out the RAM by doing it in small batches
from keras.preprocessing.image import ImageDataGenerator

imagegen = ImageDataGenerator()

# The ImageDataGenerator itself inferences the class labels and the number of classes from the folder names inside "train" folder
train = imagegen.flow_from_directory("D:/D_Documents/AI_ML/imagenette2/imagenette2/train", class_mode="categorical", shuffle=False, batch_size=128, target_size=(224, 224))

# The ImageDataGenerator itself inferences the class labels and the number of classes from the folder names inside "val" folder
val = imagegen.flow_from_directory("D:/D_Documents/AI_ML/imagenette2/imagenette2/val", class_mode="categorical", shuffle=False, batch_size=128, target_size=(224, 224))

# 2.Builing a basic CNN model for Image Classification
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, InputLayer, BatchNormalization, Dropout

# build a sequential model
model = Sequential()
model.add(InputLayer(input_shape=(224, 224, 3)))

# 1st conv block
model.add(Conv2D(25, (5, 5), activation='relu', strides=(1, 1), padding='same'))
model.add(MaxPool2D(pool_size=(2, 2), padding='same'))
# 2nd conv block
model.add(Conv2D(50, (5, 5), activation='relu', strides=(2, 2), padding='same'))
model.add(MaxPool2D(pool_size=(2, 2), padding='same'))
# Batch Normalization is a normalization technique done between the layers of a Neural Network instead of in the raw data.
# This has the impact of "settling" the learning process and drastically decreasing the number of training epochs required to train deep neural networks.
model.add(BatchNormalization())

# 3rd conv block
model.add(Conv2D(70, (3, 3), activation='relu', strides=(2, 2), padding='same'))
model.add(MaxPool2D(pool_size=(2, 2), padding='valid'))
model.add(BatchNormalization())
# ANN block
model.add(Flatten())
model.add(Dense(units=100, activation='relu'))
model.add(Dense(units=100, activation='relu'))
model.add(Dropout(0.25))

# output layer
model.add(Dense(units=10, activation='softmax'))

# compile model
model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])

# fit on data for 30 epochs
model.fit(train, epochs=10, validation_data=val)

