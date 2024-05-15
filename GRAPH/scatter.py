import sys
sys.path.append('../../ft_linear_regression')
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import csv
import TOOLS as tools

file_path = "../DATA/training.csv"
data = pd.read_csv('../DATA/data.csv')
print(data.info())

X = data['km']
y = data['price']

plt.scatter(X,y)
plt.xlabel('Mileage in km')
plt.ylabel('Price')

plt.title('Price Prediction')



if os.path.exists(file_path):
    with open(file=file_path, mode='r', encoding='utf-8') as csvfile:
        dataset = csv.reader(csvfile)
        next(dataset)
        #ordonne = np.linspace( 250000, 50000)
        theta0, theta1 = map(float, next(dataset))
        xmin, xmax = X.min(), X.max()
        ordonne = np.linspace(xmin, xmax)
        plt.plot(ordonne, theta0 + theta1 * ordonne, color='r')
        #plt.plot(ordonne,  color='r')
        #theta0 = 0
        #theta1 = 0
        #theta0, theta1 = map(float, next(dataset))
        #plt.plot(ordonne, theta0 * ordonne + theta0, color='r')

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