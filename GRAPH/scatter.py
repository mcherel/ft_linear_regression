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

X = data['km'].values.reshape(-1, 1)
y = data['price'].values.reshape(-1, 1)

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

        x_regression = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
        y_regression = theta0 + theta1 * x_regression

        plt.plot(x_regression, y_regression,  color='r', label='Régression linéaire')
        plt.scatter(X, y, label='Données', color='b')
        
        plt.legend()
        plt.title('Price Prediction')

plt.show()