# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 00:31:24 2021

@author: KHUSH
"""
from sklearn.preprocessing import StandardScaler # 
import pickle
import os

from Preprocess import driverPre
from TrainTestSplit import tts
from Model import model_maker, train_model
from Output import driver_output

def save_model(path, model, pred_col):
    if path not in os.listdir():
        os.mkdir(path)
    i=0
    model_name = f"model_{pred_col}_{i}.sav"
    while model_name in os.listdir(path):
        i=i+1
        model_name = f"model_{pred_col}_{i}.sav"
    path = os.path.join(path,model_name)
    pickle.dump(model, open(path, 'wb'))
    return path

def algoMLP (df,prediction_columns, output_path):
    x, y, bool_clas = driverPre(df, prediction_columns)
    scx = StandardScaler()
    scy = StandardScaler()
    x, y, x_train, y_train, x_test, y_test, x_val, y_val, scx, scy = tts(x,y,bool_clas,scx, scy, prediction_columns)
    model = model_maker(bool_clas)
    model = train_model(model, x_train, y_train)
    driver_output(x, x_train, x_test, y, y_train, y_test, x_val, y_val, model, scx, scy)
    save_model(output_path, model, prediction_columns)
    return None