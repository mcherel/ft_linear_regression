import csv
import random

theta0 = random.uniform(-1, 1)
theta1 = random.uniform(-1, 1)
theta0 = 0
theta1 = 0
file = "data.csv"

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

def estimated_price(theta0, theta1, X_km):
    return [theta0 + theta1 * x for x in X_km]


def hypothesis(theta0, theta1, X_km, y_price):
    learning_rate = 0.0000000001
    num_iterations = 1000
    data_points = len(X_km)

    for i in range(num_iterations):
        estimated_prices = estimated_price(theta0, theta1, X_km) 
        # Calculate gradient
        grad_theta0 = (1/data_points) * sum(estimated_prices[i] - y_price[i] for i in range(data_points))
        grad_theta1 = (1/data_points) * sum((estimated_prices[i] - y_price[i]) * X_km[i] for i in range(data_points))
        
        # Update params
        theta0 -= learning_rate * grad_theta0
        theta1 -= learning_rate * grad_theta1

        print(f"Iteration {i+1}: theta0={theta0}, theta1={theta1}")

    return theta0, theta1

X_km, y_price = open_csv(file)
theta0, theta1 = hypothesis(theta0, theta1, X_km, y_price)
print(theta0)
print(theta1)
