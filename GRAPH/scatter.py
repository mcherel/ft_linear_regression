import sys
sys.path.append('../../ft_linear_regression')
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from TOOLS import open_csv
from sklearn import metrics
from TOOLS import  estimate_price

training_data_file = "../DATA/training.csv"
file = "../DATA/data.csv"

# Check if the ddata file exists
if os.path.exists(file):
        # Loading  data into two lists
        data = pd.read_csv(file)
        X = data['km'].values.reshape(-1, 1)
        y = data['price'].values.reshape(-1, 1)

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

#Plot regression line from training data if available
if os.path.exists(training_data_file):
        theta0, theta1 = open_csv.open_csv(training_data_file)

        # Adding a red line
        xmin, xmax = X.min(), X.max()
        ordonne = np.linspace(xmin, xmax)
        plt.plot(ordonne, theta1 * ordonne + theta0, color='r', label='Linear Regression Line')

        # Prediction
        y_predict = estimate_price.hypothesis(theta0, theta1, X)

        # Print Mean Absolute Error (MAE)
        print(f'MAE: \t{metrics.mean_absolute_error(y, y_predict)}')

        # Print Mean Squared Error (MSE)
        print(f'MSE: \t{metrics.mean_squared_error(y, y_predict)}')

        # Print Root Mean Squared Error (RMSE)
        print(f'RMSE: \t{np.sqrt(metrics.mean_squared_error(y, y_predict))}')

        # Print R-squared (R²)
        print(f'R²: \t{metrics.r2_score(y, y_predict)}')


# Graph printing
plt.legend() # ensures the labels for the scatter plot and regression line are displayed.
plt.show()