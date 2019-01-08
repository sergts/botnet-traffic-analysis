#!/usr/bin/python

import sys
import os
from glob import iglob
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from keras.models import model_from_yaml
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Activation
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split


def load_data():
    print('Loading data')
    print('Loading gafgyt data')
    df_gafgyt = pd.concat((pd.read_csv(f) for f in iglob('../data/**/gafgyt_attacks/*.csv', recursive=True)), ignore_index=True)
    print('Loaded, shape: ')
    print(df_gafgyt.shape)
    df_gafgyt['class'] = 'gafgyt'
    print('Loading mirai data')
    df_mirai = pd.concat((pd.read_csv(f) for f in iglob('../data/**/mirai_attacks/*.csv', recursive=True)), ignore_index=True)
    print('Loaded, shape: ')
    print(df_mirai.shape)
    df_mirai['class'] = 'mirai'
    print('Loading benign data')
    df_benign = pd.concat((pd.read_csv(f) for f in iglob('../data/**/benign_traffic.csv', recursive=True)), ignore_index=True)
    print('Loaded, shape: ')
    print(df_benign.shape)
    df_benign['class'] = 'benign'
    df = df_benign.append(df_gafgyt.sample(n=df_benign.shape[0], random_state=17)).append(df_mirai.sample(n=df_benign.shape[0], random_state=17))
    return df


def create_model(input_dim, add_hidden_layers, hidden_layer_size):
    model = Sequential()
    model.add(Dense(hidden_layer_size, activation="tanh", input_shape=(input_dim,)))
    for i in range(add_hidden_layers):
        model.add(Dense(hidden_layer_size, activation="tanh"))
    model.add(Dense(3))
    model.add(Activation('softmax'))
    return model

def train(top_n_features = None):
    df = load_data()
    train_with_data(top_n_features, df)

def train_with_data(top_n_features = None, df = None):
    X = df.drop(columns=['class'])
    if top_n_features is not None:
        fisher = pd.read_csv('../data/top_features_fisherscore.csv')
        features = fisher.iloc[0:int(top_n_features)]['Feature'].values
        X = X[list(features)]
    Y = pd.get_dummies(df['class'])
    print('Splitting data')
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    print('Transforming data')
    scaler.fit(x_train)
    input_dim = X.shape[1]
    scalerfile = f'./models/scaler_{input_dim}.sav'
    pickle.dump(scaler, open(scalerfile, 'wb'))
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    print('Creating a model')
    
    model = create_model(input_dim, 1, 128)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    cp = ModelCheckpoint(filepath=f'./models/model_{input_dim}.h5',
                               save_best_only=True,
                               verbose=0)
    tb = TensorBoard(log_dir=f'./logs',
                histogram_freq=0,
                write_graph=True,
                write_images=True)
    epochs = 25
    model.fit(x_train, y_train,
                    epochs=epochs,
                    batch_size=256,
                    validation_data=(x_test, y_test),
                    verbose=1,
                    callbacks=[tb, cp])
    print('Model evaluation')
    print('Loss, Accuracy')
    print(model.evaluate(x_test, y_test))
    y_pred_proba = model.predict(x_test)
    y_pred = np.argmax(y_pred_proba, axis=1)
    cnf_matrix = confusion_matrix(np.argmax(y_test.values, axis=1), y_pred)
    print('Confusion matrix')
    print('benign  gafgyt  mirai')
    print(cnf_matrix)

if __name__ == '__main__':
    train(*sys.argv[1:])