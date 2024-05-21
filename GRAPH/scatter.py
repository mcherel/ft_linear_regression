import sys
sys.path.append('../../ft_linear_regression')
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import csv
import TOOLS as tools
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

training_data = "../DATA/training.csv"
data = pd.read_csv('../DATA/data.csv')
X = data['km'].values.reshape(-1, 1)
y = data['price'].values.reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)
plt.scatter(X,y)
regressor = LinearRegression()
#Training
regressor.fit(X_train, y_train)
regressor = LinearRegression()
#Training
regressor.fit(X_train, y_train)
theta11 = regressor.coef_
theta00 = regressor.intercept_
if os.path.exists(training_data):
        with open(file=training_data, mode='r', encoding='utf-8') as csvfile:
                dataset = csv.reader(csvfile)
                next(dataset)
                theta0, theta1 = map(float, next(dataset))
                theta1 = [[theta1]]
                theta0 = [theta0]

                print(f'{theta11} {theta1}')
                print(f'{theta00} {theta0}')
                # Adding a red line
                xmin, xmax = X.min(), X.max()
                ordonne = np.linspace(xmin, xmax)
                plt.plot(ordonne,theta1[0]*ordonne + theta0, color='r')

                #Linear Regression
                plt.plot(ordonne,theta11[0]*ordonne + theta00, color='k')

# Prediction from test base
y_predict = regressor.predict(X_test)

# Metrics
print(f'MAE: \t{metrics.mean_absolute_error(y_test, y_predict)}')
print(f'MSE: \t{metrics.mean_squared_error(y_test, y_predict)}')
print(f'RMSE: \t{np.sqrt(metrics.mean_squared_error(y_test, y_predict))}')
print(f'R²: \t{metrics.r2_score(y_test, y_predict)}')

plt.show()

""" training_data = "../DATA/training.csv"
data = pd.read_csv('../DATA/data.csv')

X = data['km'].values.reshape(-1, 1)
y = data['price'].values.reshape(-1, 1)

plt.xlabel('Mileage in km')
plt.ylabel('Price')

if os.path.exists(training_data):
    with open(file=training_data, mode='r', encoding='utf-8') as csvfile:
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
        plt.title('Price Prediction') """

plt.show()