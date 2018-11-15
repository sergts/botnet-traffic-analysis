import sys
import pandas as pd
import numpy as np
import random
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from glob import iglob
from sklearn.metrics import recall_score, accuracy_score, precision_score

def load_mal_data():
    df_mirai = pd.concat((pd.read_csv(f) for f in iglob('../data/**/mirai_attacks/*.csv', recursive=True)), ignore_index=True)
    df_gafgyt = pd.concat((pd.read_csv(f) for f in iglob('../data/**/gafgyt_attacks/*.csv', recursive=True)), ignore_index=True)
    return df_mirai.append(df_gafgyt)

def test():
    test_with_data(load_mal_data())

def test_with_data(df_malicious):
    print("Testing")
    df = pd.concat((pd.read_csv(f) for f in iglob('../data/**/benign_traffic.csv', recursive=True)), ignore_index=True)
    x_train, x_opt, x_test = np.split(df.sample(frac=1, random_state=17), [int(1/3*len(df)), int(2/3*len(df))])
    scaler = StandardScaler()
    scaler.fit(x_train.append(x_opt))

    print(f"Loading model")
    saved_model = load_model("models/model.h5")
    with open('threshold') as t:
        tr = np.float64(t.read())
    print(f"Calculated threshold is {tr}")
    model = AnomalyModel(saved_model, tr)

    df_benign = pd.DataFrame(x_test, columns=df.columns)
    df_benign['malicious'] = False
    df_malicious = df_malicious.sample(n=df_benign.shape[0], random_state=17)
    df_malicious['malicious'] = True
    df = df_benign.append(df_malicious)
    X_test = df.drop(columns=['malicious'])
    X_test = scaler.transform(X_test)
    Y_test = df['malicious']
    Y_pred = model.predict(X_test)
    print('Accuracy')
    print(accuracy_score(Y_test, Y_pred))
    print('Recall')
    print(recall_score(Y_test, Y_pred))
    print('Precision')
    print(precision_score(Y_test, Y_pred))

    print("---------------------------------")

class AnomalyModel:
    def __init__(self, model, threshold):
        self.model = model
        self.threshold = threshold

    def predict(self, x):
        x_pred = self.model.predict(x)
        mse = np.mean(np.power(x - x_pred, 2), axis=1)
        y_pred = mse > self.threshold
        return y_pred



if __name__ == '__main__':
    test()