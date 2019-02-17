import tensorflow as tf
import numpy as np
import csv

reader = csv.reader(open("/home/kavindu/workspace/PycharmProjects/ai/input/winequality-white.csv", "r"), delimiter=",")
train_data = list(reader)
train_data = np.array(train_data)

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(40, activation=tf.nn.relu, input_shape=[11]),
    tf.keras.layers.Dense(25, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.relu),
    tf.keras.layers.Dense(1)
])

model.summary()
model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['mse'])

model.fit(train_data[:, 0:11], train_data[:, 11:12], epochs=3000)
model.evaluate(train_data[:, 0:11], train_data[:, 11:12])
x = model.predict(train_data[:, 0:11])
print(x)

