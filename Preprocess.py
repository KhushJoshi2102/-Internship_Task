# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 00:56:03 2021

@author: KHUSH
"""
import numpy as np
import pandas as pd

def cleaning(df):              # Cleaning of Data
    df = df.dropna()           # drop the null entries in the dataframe
    df = df.drop_duplicates()  # drope the duplicate entries in the dataframe
    return df

def categorial_encoding(df,pred_col):                              # Categorial Encoding of Data
    boolean = False
    cat_c = [c for c in df.columns if df[c].dtype=='O']
    # print(cat_c)
    # print(pred_col)
    for each in pred_col:
        if each in cat_c:
            boolean = True      
            break
    from sklearn.preprocessing import LabelEncoder
    labelEn = LabelEncoder()
    for each in cat_c:
        df[each]=labelEn.fit_transform(df[each])
    from sklearn.preprocessing import OneHotEncoder 
    for each in cat_c:
        if len(np.unique(df[each].values)) > 2:
            onehotencoder = OneHotEncoder() 
            df[each]=onehotencoder.fit_transform(df[each])
    return df, boolean

def feature_selection (df,pred_col):
    cor = df.corr()
    index = pd.Index([])
    for e in pred_col:
        cor_target = abs(cor[e])
        relevant_features = cor_target[cor_target>0.5]
        index = index.union(relevant_features.index)
    df = df[index]
    return df

def df_xy(df,pred_col):                           # Function for the seperation of x and y variables
    index = df.columns
    for e in pred_col:
        dft = df.loc[ : , df.columns != e]
        index = index.intersection(dft.columns)
    dfx = df[index]
    dfy = df[pred_col]
    return dfx, dfy

def get_values(dfx,dfy):
    x = dfx.values[:,:]
    y = dfy.values[:,:]
    return x,y

def driverPre(df,prediction_columns):
    df = cleaning(df)
    df, boolean = categorial_encoding(df,prediction_columns)
    df = feature_selection(df, prediction_columns)
    dfx, dfy = df_xy(df, prediction_columns)
    x,y = get_values(dfx,dfy)
    return x, y, boolean