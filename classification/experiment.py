import train
import sys
from glob import iglob
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from keras.models import load_model
from sklearn.model_selection import train_test_split
import lime
import lime.lime_tabular

class ModelWrapper:
    def __init__(self, model, scaler):
        self.model = model
        self.scaler = scaler

    def scale_predict(self, x):
        x = self.scaler.transform(x)
        return self.model.predict(x)

df = train.load_data()

vals = [115]

test = pd.read_csv('../data/Explanation_dataset-1.csv')
fisher = pd.read_csv('../data/top_features_fisherscore.csv')

for val in vals:

    features = fisher.iloc[0:int(val)]['Feature'].values
    X = df[list(features)]
    
    Y = pd.get_dummies(df['class'])
    classes = Y.columns.tolist()
    x_train, _, y_train, _ = train_test_split(X, Y, test_size=0.2, random_state=42)
    x_test = test[list(features)]
    y_test = pd.get_dummies(test['Label'])
    scaler = pickle.load(open(f'models/scaler_{val}.sav', 'rb'))
    model = load_model(f'models/model_{val}.h5')
    wrapper = ModelWrapper(model, scaler)

    x = scaler.transform(x_test)
    print(model.evaluate(x, y_test))

    header = f'Deep Learning with {val} features'
    header2 = '#,Data Point,,' + ','.join(features) + ',,Predicted,,True'
    rows = [header+ "\n", header2+ "\n"]

    explainer = lime.lime_tabular.LimeTabularExplainer(x_train.values, feature_names=X.columns.tolist(), class_names=classes, discretize_continuous=True)
    for i in range(150):
        print(f'Explaining for record nr {i}')
        exp = explainer.explain_instance(x_test.values[i], wrapper.scale_predict, num_features=val)
        print(y_test.values[i])
        print(exp.as_list())
        l = x_test.values[i].astype(str).tolist()
        es = ','.join([a for a,b in exp.as_list()])
        row = str(i+1)+',['+ ' '.join(l) + '],,' + es + ',,' + str(np.argmax(wrapper.scale_predict(x_test.values[i].reshape(1, -1)))) + ',,' + str(np.argmax(y_test.values[i]))
        rows.append(row+ "\n")
    with open(f'exp{val}.csv', 'w') as c:
        c.writelines(rows)




    






