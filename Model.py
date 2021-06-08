# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 03:05:59 2021

@author: KHUSH
"""

def model_maker(boolean):
    if boolean:
        from sklearn.multioutput import MultiOutputClassifier
        from sklearn.neural_network import MLPClassifier
        model = MultiOutputClassifier(MLPClassifier(hidden_layer_sizes=(500,),max_iter = 10000, solver = 'sgd', learning_rate="adaptive", learning_rate_init = 0.001))
    else:
        from sklearn.multioutput import MultiOutputRegressor
        from sklearn.neural_network import MLPRegressor
        model = MultiOutputRegressor(MLPRegressor(hidden_layer_sizes=(500,),max_iter = 10000, solver = 'sgd', learning_rate="adaptive", learning_rate_init = 0.001))
    return model

def train_model(model,x_train,y_train):
    model = model.fit(x_train,y_train)
    return model

def predict(model,x):
    y = model.predict(x)
    return y