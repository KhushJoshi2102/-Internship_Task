# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 02:52:33 2021

@author: KHUSH
"""
import numpy as np
from sklearn.model_selection import train_test_split

def balance(x,y):
    from random import randrange
    s = y.shape[1]
    if s==1:
        s=2
    num = np.zeros(s)
    for each in y:
        i=0
        # print(each)
        for e in each:
            if s==2:
                if e==0:
                    num[0] = num[0] + 1
                else:
                    num[1] = num[1] + 1
            else:
                if e == 1:
                    num[i] = num[i]+1
                i=i+1
    min_=min(num)
    i=0
    for e in num:
        if e == min_:
            num = np.zeros(s).tolist()
            num[i] = 1
            num = np.asarray(num)
            break
        i=i+1
    num = np.zeros(s)
    came_index = []
    x_b=[]
    y_b=[]
    classes = np.unique(y)
    # boolean = True
    while min(num) != min_ or max(num) != min_:
        j=0
        while j<len(num):
            if num[j]<min_:
                index = randrange(y.shape[0])
                # print(num)
                if index not in came_index:
                    if (y[index] == classes[j]).all():
                        x_b.append(x[index])
                        y_b.append(y[index])
                        # print(len(y_b))
                        came_index.append(index)    
                        num[j] = num[j] + 1
            j=j+1
    x_b = np.asarray(x_b)
    y_b = np.asarray(y_b)
    return x_b,y_b

def fit_scaler(x,sc):
    sc = sc.fit(x)
    return sc

def scaling(x,sc): # Feature scaling of data
    if sc == None:
        return x
    x = sc.transform(x)
    return x

def tts(x,y,boolean,scx,scy,pred_col):
    b = True
    if not boolean:
        scy = fit_scaler(y,scy)
    else:
        scy = None
        if len(pred_col)==1:
            b = False
            x_b,y_b = balance(x,y)
    scx = fit_scaler(x,scx)
    if b:
        x_b, y_b = x, y
    x_train,x_test,y_train,y_test = train_test_split(x_b,y_b,test_size=0.2,random_state=0)
    x_train,x_val,y_train,y_val = train_test_split(x_train,y_train,test_size=0.2,random_state=0)
    y_train, y_test, y_val, y = scaling(y_train, scy), scaling(y_test, scy), scaling(y_val, scy), scaling(y,scy)
    x_train, x_test, x_val, x = scaling(x_train, scx), scaling(x_test, scx), scaling(x_val, scx), scaling(x,scx)
    return x, y, x_train, y_train, x_test, y_test, x_val, y_val, scx, scy