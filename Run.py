# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 18:17:21 2021

@author: KHUSH
"""
import pandas as pd

from Driver import algoMLP

output_path = "models"

print("Predicting diagnosis column with MLPClassifier for single class.")
df = pd.read_csv('cancer_data.csv')
prediction_columns = ["diagnosis"]
algoMLP(df, prediction_columns, output_path)

print()
print()

print("Predicting diagnosis & diagnosis1(dummy) columns with MLPClassifier for multiple classes.")
df["diagnosis1"] = df["diagnosis"].values
prediction_columns = ["diagnosis","diagnosis1"]
algoMLP(df, prediction_columns, output_path)

print()
print()

print("Predictiong radius_mean with MLPRegressor (single column prediction)")
df = pd.read_csv('C:/Users/KHUSH/Documents/cancer_data.csv')
prediction_columns = ["radius_mean"]
algoMLP(df,prediction_columns, output_path)

print()
print()
print()

print("Predictiong radius_mean and area_mean with MLPRegressor (multiple column prediction)")
prediction_columns = ["radius_mean","area_mean"]
algoMLP(df,prediction_columns, output_path)