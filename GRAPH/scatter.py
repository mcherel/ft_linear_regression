import sys
sys.path.append('../../ft_linear_regression')
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from TOOLS import open_csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
#from sklearn import metrics

training_data_file = "../DATA/training.csv"
file = "../DATA/data.csv"
if os.path.exists(file):
        data = pd.read_csv(file)
        X = data['km'].values.reshape(-1, 1)
        y = data['price'].values.reshape(-1, 1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)
        plt.scatter(X,y)
        regressor = LinearRegression()
        #Training
        regressor.fit(X_train, y_train)
        #Training
        #theta11 = regressor.coef_
        #theta00 = regressor.intercept_
else:
        print(f"There is no {file}")
        if os.path.exists(training_data_file):
            os.remove(training_data_file)
        exit()
#Plot rendering
if os.path.exists(training_data_file):
        theta0, theta1 = open_csv.open_csv(training_data_file)
        theta1 = [theta1]

        #print(f'{theta11} {theta1}')
        #print(f'{theta00} {theta0}')
        # Adding a red line
        xmin, xmax = X.min(), X.max()
        ordonne = np.linspace(xmin, xmax)
        plt.plot(ordonne,theta1[0]*ordonne + theta0, color='r')

        #Linear Regression
        #plt.plot(ordonne,theta11[0]*ordonne + theta00, color='k')

# Prediction from test base
"""y_predict = regressor.predict(X_test)"""

# Metrics
"""print(f'MAE: \t{metrics.mean_absolute_error(y_test, y_predict)}')
print(f'MSE: \t{metrics.mean_squared_error(y_test, y_predict)}')
print(f'RMSE: \t{np.sqrt(metrics.mean_squared_error(y_test, y_predict))}')
print(f'R²: \t{metrics.r2_score(y_test, y_predict)}')"""

plt.show()