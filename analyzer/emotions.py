# Setup and Imports

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt
import random

def show_history(h):
    epochs_trained = len(h.history['loss'])
    plt.figure(figsize=(16, 6))

    plt.subplot(1, 2, 1)
    plt.plot(range(0, epochs_trained), h.history.get('accuracy'), label='Training')
    plt.plot(range(0, epochs_trained), h.history.get('val_accuracy'), label='Validation')
    plt.ylim([0., 1.])
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(0, epochs_trained), h.history.get('loss'), label='Training')
    plt.plot(range(0, epochs_trained), h.history.get('val_loss'), label='Validation')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def show_confusion_matrix(y_true, y_pred, classes):
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred, normalize='true')

    plt.figure(figsize=(0, 0))
    sp = plt.subplot(1, 1, 1)
    ctx = sp.matshow(cm)
    plt.xticks(list(range(0, 6)), labels=classes)
    plt.yticks(list(range(0, 6)), labels=classes)
    plt.colorbar(ctx)
    plt.show()

print('Using Tensorflow version', tf.__version__)


# Importing Data

from datasets import load_dataset

dataset = load_dataset('emotion')

print(dataset)

train = dataset['train']
val = dataset['validation']
test = dataset['test']

def get_tweet(data):
    tweets = [x['text'] for x in data]
    labels = [x['label'] for x in data]
    return tweets, labels

tweets, labels = get_tweet(train)

print(tweets[4], labels[4])


# Tokenizer

from tensorflow.keras.preprocessing.text import Tokenizer       # type: ignore

tokenizer = Tokenizer(num_words=1000, oov_token='<UNK>')
tokenizer.fit_on_texts(tweets)
print(tokenizer.texts_to_sequences([tweets[0]]))


# Padding and Truncating Sequences

lengths = [len(t.split(' ')) for t in tweets]
# plt.hist(lengths, bins=len(set(lengths)))
# plt.show()

max_len = 50

from tensorflow.keras.preprocessing.sequence import pad_sequences      # type: ignore

def get_sequences(tokenizer, tweets):
    sequences = tokenizer.texts_to_sequences(tweets)
    padded = pad_sequences(sequences, truncating='post', padding='post', maxlen=max_len)
    return padded

padded_train_seq = get_sequences(tokenizer, tweets)
print(padded_train_seq[0])


# Preparing the labels

classes = set(labels)
print(classes)

# plt.hist(labels, bins=11)
# plt.show()

# class_to_index = dict((c, i) for i, c in enumerate(classes))
# index_to_class = dict((v, k) for k, v in class_to_index.items())

# print(class_to_index)

# names_to_ids = lambda labels: np.array([class_to_index.get(x) for x in labels])

train_labels = np.array(labels)
# print(train_labels[4])


# Creating the Model

model = keras.models.Sequential([
    keras.layers.Embedding(1000, 16),
    keras.layers.Bidirectional(keras.layers.LSTM(20, return_sequences=True)),
    keras.layers.Bidirectional(keras.layers.LSTM(20)),
    keras.layers.Dense(6, activation='softmax')
])

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.summary()


# Training the Model

val_tweets, val_labels = get_tweet(val)
val_seq = get_sequences(tokenizer, val_tweets)
val_labels = np.array(val_labels)

print(val_tweets[1], val_labels[1])

h = model.fit(
    padded_train_seq, train_labels,
    validation_data=(val_seq, val_labels),
    epochs=20,
    callbacks=[
        keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=2)
    ]
)


# Evaluating the Model

show_history(h)

test_tweets, test_labels = get_tweet(test)
test_seq = get_sequences(tokenizer, test_tweets)
test_labels = np.array(test_labels)

_ = model.evaluate(test_seq, test_labels)

i = random.randint(0, len(test_labels)-1)
print('Sentence: ', test_tweets[i])
print('Emotion: ', test_labels[i])
p = model.predict(np.expand_dims(test_seq[i], axis=0))
pred_class = np.argmax(p).astype('uint8')
print('Predicted Emotion:', pred_class)