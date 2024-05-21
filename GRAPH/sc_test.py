import sys
sys.path.append('../../ft_linear_regression')
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from TOOLS import open_csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

training_data_file = "../DATA/training.csv"
file = "../DATA/data.csv"

# Check if the data file exists
if os.path.exists(file):
    # Loading data into two lists
    data = pd.read_csv(file)
    X = data['km'].values.reshape(-1, 1)
    y = data['price'].values.reshape(-1, 1)

    # Splitting data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

    # Plot the data
    plt.scatter(X, y, label='Car prices')

    # Creating and training the model
    regressor = LinearRegression()
    regressor.fit(X, y)
else:
    print(f"There is no {file}")
    if os.path.exists(training_data_file):
        os.remove(training_data_file)
    exit()

# Plot regression line from training data if available
if os.path.exists(training_data_file):
    theta0, theta1 = open_csv.open_csv(training_data_file)

    # Adding a red line
    xmin, xmax = X.min(), X.max()
    ordonne = np.linspace(xmin, xmax)
    plt.plot(ordonne, theta1 * ordonne + theta0, color='r', label='Linear Regression Line')

# Prediction
y_predict = regressor.predict(X)

# Metrics calculation
print(f'MAE: \t{metrics.mean_absolute_error(y, y_predict)}')
print(f'MSE: \t{metrics.mean_squared_error(y, y_predict)}')
print(f'RMSE: \t{np.sqrt(metrics.mean_squared_error(y, y_predict))}')
print(f'R²: \t{metrics.r2_score(y, y_predict)}')

# Adding labels and title
plt.xlabel('Mileage (km)')
plt.ylabel('Price ($)')
plt.title('Car Price Prediction')
plt.legend()

# Graph printing
plt.show()
