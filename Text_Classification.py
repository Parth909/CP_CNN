import tensorflow as tf
from tensorflow import keras
import numpy as np

data = keras.datasets.imdb

# num_words = 10000 :- Means select the 10000 most frequently occuring words
(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=88000)

print(train_data)

# We can see that each review is in form of list of number.
# That is bcz we have associated each word with a number

word_index = data.get_word_index()

# word_index is a "dict"
# print("word_index shape", word_index)

lowest_index = 99999999
for k, v in word_index.items():
    if v < lowest_index:
        lowest_index = v

print("The lowest index is", lowest_index)

# print("word_index before", word_index.items())

# word_index.items() type is "dict_items" :- dict_items([('Physics', 67), ('Maths', 87)])
# for every Items (tuple), k :- 0th element, v :- 1st element
# for every items return a "key:value" pair
# "value" is a number increase it by count 3. bcz LOWEST INDEX STARTS FROM 1 & we are adding 4 extra key:value pairs to the dict
# word_index is in the same form as before

word_index = {k:(v+3) for k, v in word_index.items()}

word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2 # UNK = UNKNOWN
word_index["<UNUSED>"] = 3

# print("word_index after processing", word_index)

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

print("Length before padding", len(test_data[0]), len(test_data[1]))
# NOTE :- as you know No of NEURONS in the INPUT LAYER are fix
# But the review are of VARIAABLE LENGTH. This is where PADDING TAGS (<PAD>) are useful

'''

# SOLUTION :- We will set a fix number like 250. A review should be 250 words long. Any review with length beyond 250 will be `Discarded` & a review with length less than 250 will be `Padded`
# padding="post" means add padding at the end
train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding="post", maxlen=250)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding="post", maxlen=250)

print("Length after padding", len(test_data[0]), len(test_data[1]))

# return a human readable string
def decode_review(text):
    # return element from dict at index "i" position. If not found return "?"
    return " ".join([reverse_word_index.get(i, "?") for i in text])

print(decode_review(test_data[0]))
print(test_data[0])

# MODEL
model = keras.Sequential()
# This the same as adding the array of Layers in keras.Sequential()
model.add(keras.layers.Embedding(88000, 16))
# Creating 88,000 word vectors for A SINGLE WORD each in 16 Dimensional Space
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation="relu"))
model.add(keras.layers.Dense(1, activation="sigmoid"))
# sigmoid activation function will give us value between 0-1

model.summary()

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
# using "binary_crossentropy" bcz we have 2 options for the SINGLE OUTPUT NEURON

x_val = train_data[:10000]
x_train = train_data[10000:]

y_val = train_labels[:10000]
y_train = train_labels[10000:]

fitModel = model.fit(x_train, y_train, epochs=40, batch_size=512, validation_data=(x_val, y_val), verbose=1)

results = model.evaluate(test_data, test_labels)

print(results)

# Testing single data
test_review = test_data[1]
predict = model.predict([test_review])
print("Review :- ")
print(decode_review((test_review)))
print("Prediction :- ", str(predict[0]))
print("Actual :- ", str(test_labels[0]))
print(results)

# ====> SAVING KERAS MODEL
# Saving KERAS MODEL is different from what we have used earlier
model.save("saved_model.h5")
'''

# ====> LOADING THE SAVED MODEL

def review_encode(word_list):
    encoded = [1]
    for word in word_list:
        if word.lower() in word_index:
            # add that number in list
            encoded.append(word_index[word.lower()])
        else:
            encoded.append(2)
    return encoded

model = keras.models.load_model("saved_model.h5")

with open("negative_review.txt", encoding="utf-8") as f:
    for line in f.readlines():
        nline = line.replace(",", "").replace(".", "").replace("(", "").replace(")", "").replace(":", "") \
                .replace("\"", "").strip().split(" ")
        encode = review_encode(nline)
        # pad_sequences is expecting a list of list
        encode = keras.preprocessing.sequence.pad_sequences([encode], value=word_index["<PAD>"], padding="post", maxlen=250)
        prediction = model.predict(encode)
        print("line :- ", line)
        print("encode :- ", encode)
        print("prediction :- ", prediction)

