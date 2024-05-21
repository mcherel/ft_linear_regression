import sys
sys.path.append('../../ft_linear_regression')
import csv
import os
from TOOLS import open_csv, estimate_price
import numpy as np



def gradient_descent(theta0, theta1, X_km, y_price):
    learning_rate = 0.0000001
    num_iterations = 10000
    data_points = len(X_km)


    for i in range(num_iterations):
        #Prediction
        predictions = estimate_price.hypothesis(theta0, theta1, X_km)
        
        # Calculate gradient
        error = np.array(predictions) - np.array(y_price)
        theta0 -= learning_rate * (1/data_points) * sum(error[j] - y_price[j] for j in range(data_points))
        theta1 -= learning_rate * (1/data_points) * sum((predictions[j] - y_price[j]) * X_km[j] for j in range(data_points))

        #print(f"Iteration {i+1}: theta0={theta0}, theta1={theta1}")

    return theta0, theta1

def main():
    file = "../DATA/data.csv"
    training_data_file = "../DATA/training.csv"
    if os.path.exists(file):
        X_km, y_price = open_csv.open_csv(file)
        #Normalizing the data
        X_km_mean = np.mean(X_km)
        X_km_std = np.std(X_km)
        X_km = (X_km - X_km_mean) / X_km_std

        y_price_mean = np.mean(y_price) #mean prie
        y_price_std = np.std(y_price) #price deviation
        y_price = (y_price - y_price_mean) / y_price_std
        
        #initializing thetas to random
        theta0 = 1 #initialized with  1 to get positive theta0
        theta1 = -0.8  #initialized with negative number to get negative coefficient
        theta0, theta1 = gradient_descent(theta0, theta1, X_km, y_price)

        #Denormalizing the data
        theta0 *= y_price_std / X_km_std + y_price_mean - theta1 * (y_price_std * X_km_mean / X_km_std)
        theta1 *= y_price_std / X_km_std
        
        # Creating thraining.csv containing theta0 and theta1
        with open(training_data_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['theta0', 'theta1'])
            writer.writerow([theta0, theta1])
        print(f'theta0 : {theta0}')
        print(f'theta1 : {theta1}')
        print(f'File {training_data_file} created')
    else:
        print(f"There is no {file}")
        if os.path.exists(training_data_file):
            os.remove(training_data_file)
        exit()




if __name__ == '__main__':
    main()