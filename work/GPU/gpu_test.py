#https://gist.github.com/raulqf/2d5f2b33549e56a6bb7c9f52a7fd471c
from __future__ import absolute_import, division, print_function, unicode_literals
from pip import main

import tensorflow as tf
import os

print(f'tensorflow version: {tf.__version__}')

# tf.debugging.set_log_device_placement(True)``
print(f'gpu device name: {tf.test.gpu_device_name()}')
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# import intel_extension_for_tensorflow as itex
# print(itex.__version__)

import tensorflow as tf

def run_test():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=5)
    model.evaluate(x_test,  y_test, verbose=2)

if __name__ == '__main__':
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        with tf.device('/GPU:0'):
            print('Using GPU: ')
            run_test()
    
    with tf.device('/CPU'):
        print('Using CPU: ')
        run_test()
    
