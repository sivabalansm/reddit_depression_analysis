import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import tensorflow as tf
import keras
from keras import layers

model = keras.Sequential()
model.add(layers.SimpleRNN(64, input_shape=(None, 28)))
model.add(layers.BatchNormalization())
model.add(layers.Dense(10))

mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train/255.0, x_test/255.0
sample, sample_label = x_test[0], y_test[0]

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer="sgd",
    metrics=["accuracy"],
)

model.fit(
    x_train, y_train, validation_data=(x_test, y_test), batch_size=64, epochs=1
)

result = tf.argmax(model.predict(tf.expand_dims(sample, 0)), axis=1)
print(result.numpy(), sample_label)