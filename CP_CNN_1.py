# 1. Loading images without maxing out the RAM by doing it in small batches
from keras.preprocessing.image import ImageDataGenerator
import pickle

imagegen = ImageDataGenerator()

train = imagegen.flow_from_directory("D:/D_Documents/NudeNet_Classifier_train_data_x320/nude_sexy_safe_v1_x320/training", class_mode="categorical", shuffle=False, batch_size=150, target_size=(224, 224))

val = imagegen.flow_from_directory("D:/D_Documents/NudeNet_Classifier_train_data_x320/nude_sexy_safe_v1_x320/validation", class_mode="categorical", shuffle=False, batch_size=150, target_size=(224, 224))


from tensorflow.keras.applications import VGG16

# include_top is set to False to remove softmax layer
pretrained_model = VGG16(include_top=False, weights='imagenet')
pretrained_model.summary()

from tensorflow.keras.utils import to_categorical
# extract train and val features
vgg_features_train = pretrained_model.predict(train)
vgg_features_val = pretrained_model.predict(val)
# OHE target column
train_target = to_categorical(train.labels)
val_target = to_categorical(val.labels)

from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, InputLayer, BatchNormalization, Dropout

print(vgg_features_train)

model = Sequential()
# Input shape can be anything not necessary for it to be input shape of the image
model.add(Flatten(input_shape=(7,7,512)))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.5))
model.add(BatchNormalization())
model.add(Dense(3, activation='softmax'))

# compile the model
model.compile(optimizer='adam', metrics=['accuracy'], loss='categorical_crossentropy')

model.summary()

# train model using features generated from VGG16 model
# training :- features, labels                                         testing :- features, labels
model.fit(vgg_features_train, train_target, epochs=50, batch_size=150, validation_data=(vgg_features_val, val_target))

with open("cnn_model.h5", "wb") as f:
    pickle.dump(model, f)