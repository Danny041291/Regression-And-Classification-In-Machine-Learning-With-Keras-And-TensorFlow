import os
import sys
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from keras import backend as K
from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.datasets import fashion_mnist


def remove_file(file):
    if os.path.isfile(file):
        os.remove(file)


def load_dataset():
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    train_images = train_images / 255.0
    test_images = test_images / 255.0
    return (train_images, train_labels), (test_images, test_labels), class_names


def define_model(image_dim):
    model = Sequential([
        Flatten(input_shape=image_dim),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    if os.path.isfile('classification_weights.h5'):
        model.load_weights('classification_weights.h5')
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def train_model(model, train_images, train_labels, epochs, batch_size):
    train = model.fit(train_images, train_labels, epochs=epochs, batch_size=batch_size)
    remove_file('classification_weights.h5')
    model.save('classification_weights.h5')
    return train


def evaluate_model(model, test_images, test_labels):
    return model.evaluate(test_images, test_labels, verbose=2)


def predict(model, test_images, test_labels, class_names):
    predictions = model.predict(test_images)
    test_index = random.randrange(0, len(predictions)-1, 1)
    return class_names[np.argmax(predictions[test_index])], class_names[test_labels[test_index]]


if __name__ == '__main__':
    with tf.Session() as sess:
        K.set_session(sess)
        if sys.argv[1] == 'train':
            remove_file('classification_weights.h5')
            (train_images, train_labels), (test_images, test_labels), class_names = load_dataset()
            img = test_images[1]
            model = define_model(img.shape)
            train = train_model(model, train_images, train_labels, 10, 10)
            test_loss, test_acc = evaluate_model(model, test_images, test_labels)
            print('Test loss:', test_loss)
            print('Test accuracy:', test_acc)
            plt.plot(train.history['loss'])
            plt.plot(train.history['acc'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['loss', 'acc'], loc='upper left')
            plt.show()
        elif sys.argv[1] == 'predict':
            (train_images, train_labels), (test_images, test_labels), class_names = load_dataset()
            img = test_images[1]
            model = define_model(img.shape)
            prediction, expected = predict(model, test_images, test_labels, class_names)
            print('Expected: ' + expected + ', has: ' + prediction)