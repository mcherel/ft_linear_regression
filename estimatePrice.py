import csv
#import graph


theta0 = 0
theta1 = 1
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


def hypothesis(theta0, theta1):
    X_km, y_price = open_csv(file)
    print(X_km)
    print(y_price)

    learning_rate = 0.01
    num_iterations = 1000
    data_points = len(X_km)

    # gradient descent
    for _ in range(num_iterations):
        # calculate gradient
        grad_theta0 = (1/data_points)*sum(estimated_price(theta0, theta1, X_km[i]) - y_price[i] for i in range(data_points))
        grad_theta1 = (1/data_points)*sum((estimated_price(theta0, theta1, X_km[i])*X_km[i] - y_price[i]) for i in range(data_points))
        
        # Update params
        theta0-= learning_rate*grad_theta0
        theta1-= learning_rate*grad_theta1

    return theta0, theta1

def estimated_price(theta0, theta1, X_km):
    return theta0 + theta1 * X_km
theta0, theta1 = hypothesis(theta0, theta1)
print(theta0)
print(theta1)