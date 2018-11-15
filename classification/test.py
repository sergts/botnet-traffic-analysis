import train
import sys
from glob import iglob
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from keras.models import load_model
from sklearn.model_selection import train_test_split
import lime
import lime.lime_tabular

def test(num_features, model_name):
    df = train.load_data()
    test_with_data(num_features, model_name, df)

def test_with_data(num_features, model_name, df):
    X = df.drop(columns=['class'])
    if num_features is not None:
        fisher = pd.read_csv('../fisher.csv')
        features = fisher.iloc[0:int(num_features)]['Feature'].values
        X = X[list(features)]
    Y = pd.get_dummies(df['class'])
    classes = Y.columns.tolist()
    print('Splitting data')
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    print('Transforming data')
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    model = load_model(f'models/{model_name}')

    print('Model evaluation')
    print('Loss, accuracy')
    print(model.evaluate(x_test, y_test))
    y_pred_proba = model.predict(x_test)
    y_pred = np.argmax(y_pred_proba, axis=1)

    cnf_matrix = confusion_matrix(np.argmax(y_test.values, axis=1), y_pred)
    print('Confusion matrix')
    print(classes)

    print(cnf_matrix)

    print('Explaining data using lime')
    explainer = lime.lime_tabular.LimeTabularExplainer(x_train, feature_names=X.columns.tolist(), class_names=classes, discretize_continuous=True)
    for j in range(10):
        i = np.random.randint(0, x_test.shape[0])
        print(f'Explaining for record nr {i}')
        exp = explainer.explain_instance(x_test[i], model.predict, num_features=int(num_features), top_labels=3)
        exp.save_to_file(f'lime/explanation{j}.html')
        print(exp.as_list())



if __name__ == '__main__':
    test(*sys.argv[1:])