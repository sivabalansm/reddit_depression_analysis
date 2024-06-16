# Setup and Imports

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import keras
import numpy as np
import matplotlib.pyplot as plt
import random
import json

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


# Importing Data

with open('../data/total_posts.json', 'r') as file:
    dataset = json.load(file)

random.shuffle(dataset)
len_posts = len(dataset)

train = dataset[:int(len_posts*0.8)]
val = dataset[int(len_posts*0.8):int(len_posts*0.9)]
test = dataset[int(len_posts*0.9):]

def get_tweet(data):
    tweets = [x['text'] for x in data]
    labels = [x['label'] for x in data]
    return tweets, labels

tweets, labels = get_tweet(train)


# Tokenizer

from tensorflow.keras.preprocessing.text import Tokenizer       # type: ignore

tokenizer = Tokenizer(num_words=1000, oov_token='<UNK>')
tokenizer.fit_on_texts(tweets)


# Padding and Truncating Sequences

lengths = [len(t.split(' ')) for t in tweets]
# plt.hist(lengths, bins=len(set(lengths)))
# plt.show()

max_len = 200

from tensorflow.keras.preprocessing.sequence import pad_sequences      # type: ignore

def get_sequences(tokenizer, tweets):
    sequences = tokenizer.texts_to_sequences(tweets)
    padded = pad_sequences(sequences, truncating='post', padding='post', maxlen=max_len)
    return padded

padded_train_seq = get_sequences(tokenizer, tweets)


# Preparing the labels

train_labels = np.array(labels)


# Creating the Model

model = keras.models.Sequential([
    keras.layers.Embedding(1000, 16),
    keras.layers.Bidirectional(keras.layers.LSTM(20, return_sequences=True)),
    keras.layers.Bidirectional(keras.layers.LSTM(20)),
    keras.layers.Dense(2, activation='sigmoid')
])

model.compile(  
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# model.summary()


# Training the Model

from sklearn.utils.class_weight import compute_class_weight

val_tweets, val_labels = get_tweet(val)
val_seq = get_sequences(tokenizer, val_tweets)
val_labels = np.array(val_labels)


class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)
class_weight_dict = dict(enumerate(class_weights))

print(class_weight_dict)

h = model.fit(
    padded_train_seq, train_labels,
    validation_data=(val_seq, val_labels),
    epochs=20,
    class_weight=class_weight_dict,
    callbacks=[
        keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=2)
    ]
)


# Evaluating the Model

# show_history(h)

test_tweets, test_labels = get_tweet(test)
test_seq = get_sequences(tokenizer, test_tweets)
test_labels = np.array(test_labels)

_ = model.evaluate(test_seq, test_labels)

y_pred = model.predict(test_seq)
y_pred = (y_pred > 0.5).astype(int)

y_pred = np.array(list(map(lambda x: x[1], y_pred)))


i = random.randint(0, len(test_labels)-1)
print('Sentence: ', test_tweets[i])
print('Emotion: ', test_labels[i])
p = model.predict(np.expand_dims(test_seq[i], axis=0))
pred_class = np.argmax(p).astype('uint8')
print('Predicted Emotion:', pred_class)
print(f"Confidence score: {p[0][0] * 100:.2f}%")


from sklearn.metrics import confusion_matrix

tn, fp, fn, tp = confusion_matrix(test_labels, y_pred).ravel()

fpr = fp / (fp + tn)
print(f"False Positive Rate: {fpr:.4f}")

fnr = fn / (fn + tp)
print(f"False Negative Rate: {fnr:.4f}")