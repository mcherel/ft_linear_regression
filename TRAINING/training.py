import sys
sys.path.append('../../ft_linear_regression')
import csv
import random
import ESTIMATE_PRICE.estimate_price as price

""" theta0 = random.uniform(-1, 1)
theta1 = random.uniform(-1, 1) """
theta0 = 0
theta1 = 0
file = "../DATA/data.csv"
training_data_file = "../DATA/training.csv"

def open_csv(file):
    X = []
    y = []
    with open(file=file, mode='r', encoding='utf-8') as csvfile:
        dataset = csv.reader(csvfile)
        next(dataset)
        for km, price in dataset:
            X.append(float(km))
            y.append(float(price))
    return X, y
def gradient_descent(theta0, theta1, X_km, y_price):
    learning_rate = 0.0000000001
    num_iterations = 1000
    data_points = len(X_km)

    for _ in range(num_iterations):
        estimated_prices = price.hypothesis(theta0, theta1, X_km) 
        # Calculate gradient
        error = [estimated_prices[j] - y_price[j] for j in range(data_points)]
        grad_theta0 = (1/data_points) * sum(error[j] - y_price[j] for j in range(data_points))
        grad_theta1 = (1/data_points) * sum((estimated_prices[j] - y_price[j]) * X_km[j] for j in range(data_points))
        
        # Update params
        theta0 -= learning_rate * grad_theta0
        theta1 -= learning_rate * grad_theta1

        #print(f"Iteration {i+1}: theta0={theta0}, theta1={theta1}")

    return theta0, theta1

X_km, y_price = open_csv(file)
theta0, theta1 = gradient_descent(theta0, theta1, X_km, y_price)

with open(training_data_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['theta0', 'theta1'])
    writer.writerow([theta0, theta1])


print(theta0)
print(theta1)