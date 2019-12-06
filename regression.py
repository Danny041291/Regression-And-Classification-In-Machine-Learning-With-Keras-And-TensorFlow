import os
import sys
import random
import pandas
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(1)

from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib


def remove_file(file):
    if os.path.isfile(file):
        os.remove(file)


def check_dataset(name):
    dataset = pandas.read_csv(name)
    dataset.describe(include='all')
    return dataset


def load_dataset(name):
    dataset = pandas.read_csv(name)
    X_length = len(dataset.columns)-1
    X = dataset.iloc[:, 0:X_length].values
    y = dataset.iloc[:, X_length].values.reshape(-1, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    X_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()
    X_scaled = X_scaler.fit_transform(X_train)
    y_scaled = y_scaler.fit_transform(y_train)
    remove_file('regression_X_scaler.pkl')
    remove_file('regression_y_scaler.pkl')
    joblib.dump(X_scaler, 'regression_X_scaler.pkl')
    joblib.dump(y_scaler, 'regression_y_scaler.pkl')
    return (X_scaled, y_scaled), (X_test, y_test), X_length


def define_model(X_length):
    model = Sequential([
        Dense(X_length, kernel_initializer='normal', activation='relu', input_shape=(X_length,)),
        Dense(1, kernel_initializer='normal')
    ])
    optimizer = Adam(lr=0.001)
    if os.path.isfile('regression_weights.h5'):
        model.load_weights('regression_weights.h5')
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy', 'mean_absolute_error', 'mean_squared_error'])
    return model


def train_model(model, X_train, y_train, epochs, batch_size):
    train = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
    remove_file('regression_weights.h5')
    model.save('regression_weights.h5')
    return train


def evaluate_model(model, X_test, y_test):
    X_scaler = joblib.load('regression_X_scaler.pkl')
    y_scaler = joblib.load('regression_y_scaler.pkl')
    X_transform = X_scaler.transform(X_test)
    y_transform = y_scaler.transform(y_test)
    return model.evaluate(X_transform, y_transform, verbose=2)


def predict(model, X_test, y_test):
    test_index = random.randrange(0, len(X_test) - 1, 1)
    X = np.array(X_test[test_index]).reshape(1, -1)
    y = np.array(y_test[test_index]).reshape(1, -1)
    X_scaler = joblib.load('regression_X_scaler.pkl')
    y_scaler = joblib.load('regression_y_scaler.pkl')
    X_transform = X_scaler.transform(X)
    y_transform = y_scaler.transform(y)
    result = model.predict(X_transform)
    return y_scaler.inverse_transform(result), y_scaler.inverse_transform(y_transform)


if __name__ == '__main__':
    with tf.Session() as sess:
        K.set_session(sess)
        if sys.argv[1] == 'check':
            dataset = check_dataset('regression_data.csv')
            sns.pairplot(dataset)
            plt.show()
        elif sys.argv[1] == 'train':
            remove_file('regression_weights.h5')
            (X_train, y_train), (X_test, y_test), X_length = load_dataset('regression_data.csv')
            model = define_model(X_length)
            train = train_model(model, X_train, y_train, 100, 5)
            evaluation = evaluate_model(model, X_test, y_test)
            print('Test loss:', evaluation[0])
            print('Test accuracy:', evaluation[1])
            plt.plot(train.history['loss'])
            plt.plot(train.history['acc'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['loss', 'acc'], loc='upper left')
            plt.show()
        elif sys.argv[1] == 'predict':
            (X_train, y_train), (X_test, y_test), X_length = load_dataset('regression_data.csv')
            model = define_model(X_length)
            prediction, expected = predict(model, X_test, y_test)
            print('Expected: ' + str(expected[0][0]) + ', has: ' + str(prediction[0][0]))