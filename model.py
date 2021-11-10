import pickle
import numpy as np
import cv2 as cv

shape = 128

pickle_in = open('X.pickle', "rb")
X = pickle.load(pickle_in)
pickle_in.close()

pickle_in = open('y.pickle', "rb")
y = pickle.load(pickle_in)
pickle_in.close()

for i in range(0, 20):
    cv.imshow("asd", X[i + 500])
    print(y[i + 500])
    cv.waitKey(0)

from tensorflow import keras

model = keras.Sequential()

model.add(keras.layers.BatchNormalization(input_shape=(shape, shape, 3)))
model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(keras.layers.MaxPooling2D(pool_size=2, strides=2))

model.add(keras.layers.Conv2D(6, (3, 3), activation='relu', padding='same'))
model.add(keras.layers.MaxPooling2D(pool_size=2, strides=2))

model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(keras.layers.MaxPooling2D(pool_size=2, strides=2))

model.add(keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(keras.layers.MaxPooling2D(pool_size=2, strides=2))

model.add(keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(keras.layers.MaxPooling2D(pool_size=2, strides=2))

model.add(keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(keras.layers.MaxPooling2D(pool_size=2, strides=2))

model.add(keras.layers.Flatten())

model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dense(256, activation='relu'))

model.add(keras.layers.Dense(6, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=50, )

print(model.summary())

model.save('HandGestureModel_retrained')
