# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 12:05:11 2021

@author: KHUSH
"""
import matplotlib.pyplot as plt
from Model import predict
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

def normalize_p(x):
    xmax, xmin = x.max(), x.min()
    x = (x - xmin)/(xmax - xmin)
    return x

def reshape(x):
    if len(x.shape) > 1:
        # x = x.reshape(x.shape[1],-1)
        x = [sum(e) for e in x]
        x = np.asarray(x)
        x = normalize_p(x)
    return x

def inverse_transform(x,x_train,x_test,x_val,sc):
    x = sc.inverse_transform(x)
    l = [x,x_train,x_test,x_val]
    l = [sc.inverse_transform(e) for e in l]
    # for i in range(len(l)):
    #     if l[i].shape[1] == 1:
    #     l[i] = l[i].reshape(-1)
    return l

def corrcoef(x,x_pred):
    if len(x.shape) > 1:
        corr = 0
        for i in range(x.shape[1]):
           corr = corr + np.corrcoef(x.reshape(x.shape[1],x.shape[0])[i],x_pred.reshape(x_pred.shape[1],x_pred.shape[0])[i])[0][1]
        corr = corr/x.shape[1]
    else:
        corr = np.corrcoef(x,x_pred)[0][1]
    return corr

def conf_matr(x,x_pred):
    if len(x.shape) > 1:
        conf = np.asarray([[0,0],[0,0]])
        for i in range(x.shape[1]):
              conf = conf +  confusion_matrix(x.reshape(x.shape[1],x.shape[0])[i],x_pred.reshape(x_pred.shape[1],x_pred.shape[0])[i]) 
        conf = conf/x.shape[1]
        conf = conf.astype(int)
    else:
        conf = confusion_matrix(x,x_pred)
    return conf

def plotify(x,y,pred):
    x, x_train, x_test, x_val = x
    y, y_train,y_test,y_val = y
    y_pred, y_train_pred,y_test_pred,y_val_pred = pred
    boolean = True
    for i in range(y.shape[1]):
        if boolean:
            plt.scatter(x_train,y_train[:,i], c='b', label='Orignal')
            plt.scatter(x_train,y_train_pred[:,i], c='r', label = 'Predicted')
            boolean = False
        else:
            plt.scatter(x_train,y_train[:,i], c='b')
            plt.scatter(x_train,y_train_pred[:,i], c='r')
    plt.legend()
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Training Data")
    plt.show()
    boolean = True
    for i in range(y.shape[1]):
        if boolean:
            plt.scatter(x_test,y_test[:,i], c='b', label='Orignal')
            plt.scatter(x_test,y_test_pred[:,i], c='r', label='Predicted')
            boolean = False
        else:
            plt.scatter(x_test,y_test[:,i], c='b')
            plt.scatter(x_test,y_test_pred[:,i], c='r')
    plt.legend()
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Testing Data")
    plt.show()
    boolean = True
    for i in range(y.shape[1]):
        if boolean:
            plt.scatter(x_val,y_val[:,i], c='b', label='Orignal')
            plt.scatter(x_val,y_val_pred[:,i], c='r', label='Predicted')
            boolean = False
        else:
            plt.scatter(x_val,y_val[:,i], c='b')
            plt.scatter(x_val,y_val_pred[:,i], c='r')
    plt.legend()
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Validation Data")
    plt.show()
    boolean = True
    for i in range(y.shape[1]):
        if boolean:
            plt.scatter(x,y[:,i], c='b', label='Orignal')
            plt.scatter(x,y_pred[:,i], c='r', label='Predicted')
            boolean = False
        else:
            plt.scatter(x,y[:,i], c='b')
            plt.scatter(x,y_pred[:,i], c='r')
    plt.legend()
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Overall Data")
    plt.show()
    return None

def class_reg(x,x_train,x_test,y,y_train,y_test,x_val,y_val,model, scx, scy):
    y_train_pred = predict(model,x_train)
    y_test_pred = predict(model,x_test)
    y_val_pred = predict(model,x_val)
    y_pred = predict(model,x)
    
    x,x_train,x_test,x_val = inverse_transform(x,x_train,x_test,x_val,scx)
    y, y_train,y_test,y_val = inverse_transform(y,y_train,y_test,y_val,scy)
    y_pred, y_train_pred,y_test_pred,y_val_pred = inverse_transform(y_pred,y_train_pred,y_test_pred,y_val_pred,scy)
    
    print(f"Corelation Coefficient on training data is {corrcoef(y_train,y_train_pred) : .6f}")
    print(f"Corelation Coefficient on testing data is {corrcoef(y_test,y_test_pred) : .6f}")
    print(f"Corelation Coefficient on validation data is {corrcoef(y_val,y_val_pred) : .6f}")
    print(f"Corelation Coefficient on overall data is {corrcoef(y,y_pred): .6f}")
    
    x, x_train, x_test, x_val = reshape(x), reshape(x_train), reshape(x_test), reshape(x_val)
    
    x = [x, x_train, x_test, x_val]
    y = [y, y_train,y_test,y_val]
    pred = [y_pred, y_train_pred,y_test_pred,y_val_pred]
    plotify(x,y,pred)
    return None

def class_clas(x,x_train,x_test,y,y_train,y_test,x_val,y_val,model, scx):
    y_train_pred = predict(model,x_train)
    y_test_pred = predict(model,x_test)
    y_val_pred = predict(model,x_val)
    y_pred = predict(model,x)
    
    x,x_train,x_test,x_val = inverse_transform(x,x_train,x_test,x_val,scx)
    print(f"f1 score on training data is {f1_score(y_train,y_train_pred, average = 'weighted') : .6f}")
    print(f"f1 score on testing data is {f1_score(y_test,y_test_pred, average = 'weighted') : .6f}")
    print(f"f1 score on validation data is {f1_score(y_val,y_val_pred, average = 'weighted') : .6f}")
    print(f"f1 score on overall data is {f1_score(y,y_pred, average = 'weighted') : .6f}")
    
    print(f"Confusion matrix for training data is \n{conf_matr(y_train,y_train_pred)}\n")
    print(f"Confusion matrix for testing data is \n{conf_matr(y_test,y_test_pred)}\n")
    print(f"Confusion matrix for validation data is \n{conf_matr(y_val,y_val_pred)}\n")
    print(f"Confusion matrix for overall data is \n{conf_matr(y,y_pred)}\n")
    return None

def driver_output(x,x_train,x_test,y,y_train,y_test,x_val,y_val,model, scx, scy):
    if scy == None:
        class_clas(x,x_train,x_test,y,y_train,y_test,x_val,y_val,model, scx)
    else:
        class_reg(x,x_train,x_test,y,y_train,y_test,x_val,y_val,model, scx, scy)
    return None