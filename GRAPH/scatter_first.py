
import sys
sys.path.append('../../ft_linear_regression')
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
#from sklearn import metrics
from sklearn import metrics
from TOOLS import  estimate_price, open_csv
import pandas as pd
import os
import numpy as np

training_data_file = "../DATA/training.csv"
file = "../DATA/data.csv"

# Check if the ddata file exists
if os.path.exists(file):
        # Loading  data into two lists
        data = pd.read_csv(file)
        X = data['km'].values.reshape(-1, 1)
        y = data['price'].values.reshape(-1, 1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)
        plt.scatter(X,y)
        regressor = LinearRegression()
        #Training
        regressor.fit(X_train, y_train)
        #Training
        theta11 = regressor.coef_
        theta00 = regressor.intercept_

        #Plot the data
        plt.scatter(X, y, label='Car prices')
        plt.xlabel('Mileage (km)') # adds a label to the x-axis
        plt.ylabel('Price ($)') # adds a label to the y-axis.
        plt.title('Car Price Prediction') # adds a title to the plot.

else:
        print(f"There is no {file}")
        if os.path.exists(training_data_file):
            os.remove(training_data_file)
        exit()
#Plot rendering

#Plot regression line from training data if available
if os.path.exists(training_data_file):
        theta0, theta1 = open_csv.open_csv(training_data_file)
        theta1 = [theta1]

        #print(f'{theta11} {theta1}')
        #print(f'{theta00} {theta0}')
        # Adding a red line
        xmin, xmax = X.min(), X.max()
        ordonne = np.linspace(xmin, xmax)
        plt.plot(ordonne,theta1[0]*ordonne + theta0, color='r')
        plt.plot(ordonne, theta1 * ordonne + theta0, color='r', label='Linear Regression Line')

        # Prediction
        y_predict = estimate_price.hypothesis(theta0, theta1, X)

        # Print Mean Absolute Error (MAE)
        print(f'MAE: \t{metrics.mean_absolute_error(y, y_predict)}')

        # Print Mean Squared Error (MSE)
        print(f'MSE: \t{metrics.mean_squared_error(y, y_predict)}')

        #Linear Regression
        plt.plot(ordonne,theta11[0]*ordonne + theta00, color='k')
        # Print Root Mean Squared Error (RMSE)
        print(f'RMSE: \t{np.sqrt(metrics.mean_squared_error(y, y_predict))}')

        # Print R-squared (R²)
        print(f'R²: \t{metrics.r2_score(y, y_predict)}')

# Prediction from test base
y_predict = regressor.predict(X_test)

# Metrics
print(f'MAE: \t{metrics.mean_absolute_error(y_test, y_predict)}')
print(f'MSE: \t{metrics.mean_squared_error(y_test, y_predict)}')
print(f'RMSE: \t{np.sqrt(metrics.mean_squared_error(y_test, y_predict))}')
print(f'R²: \t{metrics.r2_score(y_test, y_predict)}')

# Graph printing
plt.legend() # ensures the labels for the scatter plot and regression line are displayed.
plt.show()