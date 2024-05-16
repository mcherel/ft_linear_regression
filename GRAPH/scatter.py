import sys
sys.path.append('../../ft_linear_regression')
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import csv
import TOOLS as tools
from sklearn.preprocessing import MinMaxScaler

file_path = "../DATA/training.csv"
data = pd.read_csv('../DATA/data.csv')

print(data.info())

X = data['km'].values.reshape(-1, 1)
y = data['price'].values.reshape(-1, 1)

print(X)
print()
print(y)

plt.xlabel('Mileage in km')
plt.ylabel('Price')





if os.path.exists(file_path):
    with open(file=file_path, mode='r', encoding='utf-8') as csvfile:
        dataset = csv.reader(csvfile)
        next(dataset)
        theta0, theta1 = map(float, next(dataset))

        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        y_scaled = scaler.fit_transform(y)
        #x_regression = np.array([0, 250000])
        x_regression = np.linspace((X.min()), (X.max()), 100).reshape(-1, 1)
        y_regression = theta0 + theta1 * x_regression
        """ x_regression = np.linspace((X.min()), (X.max()), 100)
        y_regression = theta0 + theta1 * x_regression """
        print(x_regression)
        print(        )
        print(y_regression)
        #model = LinearRegression()
        #mode.fit(x_regression)
        plt.plot(x_regression, y_regression,  color='r', label ='lr')
        plt.scatter(np.array(X),np.array(y), label ='data')
        #plt.scatter(X_scaled,y_scaled, label ='data')
        plt.legend()
        plt.title('Price Prediction')
        #sorted_indices = np.argsort(X)
        #X_sorted = X[sorted_indices]
        #plt.plot(ordonne, theta1*ordonne + theta0, color='r')
        #plt.plot(X_sorted, theta1*X_sorted + theta0, color='r')

        #Data Normalizing """
        """ scaler = MinMaxScaler()
        X_normalized = scaler.fit_transform(np.array(X).reshape(-1, 1))
        plt.plot(X_normalized, theta0 + theta1 * X, color='r') """
    '''with open(file=file_path, mode='r', encoding='utf-8') as csvfile:
        dataset = csv.reader(csvfile)
        next(dataset)
        #ordonne = np.linspace( 250000, 50000)
        theta0, theta1 = map(float, next(dataset))
        #xmin, xmax = X.min(), X.max()
        #ordonne = np.linspace(xmin, xmax)
        #plt.plot(ordonne, theta0 + theta1 * ordonne, color='r')
        X_normalized = (X - X.min()) / (X.max() - X.min())
        ordonne = np.linspace((X - X.min()), (X.max() - X.min()))
        plt.plot(ordonne, theta0 + theta1 * ordonne, color='r')
        #plt.plot(ordonne,  color='r')
        #theta0 = 0
        #theta1 = 0
        #theta0, theta1 = map(float, next(dataset))
        #plt.plot(ordonne, theta0 * ordonne + theta0, color='r')'''
    pass

plt.show()

"""
file_path = "../DATA/training.csv"
theta0 = 0
theta1 = 0

if os.path.exists(file_path):
    with open(file=file_path, mode='r', encoding='utf-8') as csvfile:
        dataset = csv.reader(csvfile)
        next(dataset)
        theta0, theta1 = map(float, next(dataset)
        """