import sys
sys.path.append('../../ft_linear_regression')
import csv
# import random
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

def main():
    X_km, y_price = open_csv(file)
    theta0, theta1 = gradient_descent(theta0, theta1, X_km, y_price)

    with open(training_data_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['theta0', 'theta1'])
        writer.writerow([theta0, theta1])
    print(theta0)
    print(theta1)

if __name__ == '__main__':
    main()

'''
def cross_validation(X, y, learningRate, iterations, num_splits=5):
    mae_scores = []
    mse_scores = []
    rmse_scores = []
    r2_scores = []  # Corrected variable name
    for _ in range(num_splits):
        # Splitting the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None)
        
        # Training the model using gradient descent
        t0, t1, _, _, _ = gradientDescent(X_train, y_train, learningRate, iterations)
        
        # Making predictions on the test set
        y_pred = [t1 * normalizeElem(X_train, x) + t0 for x in X_test]
        
        # Calculating evaluation metrics
        mae = metrics.mean_absolute_error(y_test, y_pred)
        mae_scores.append(mae)
        
        mse = metrics.mean_squared_error(y_test, y_pred)
        mse_scores.append(mse)
        
        rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
        rmse_scores.append(rmse)
        
        r2 = metrics.r2_score(y_test, y_pred)  # Corrected variable name
        r2_scores.append(r2)
    return sum(mae_scores) / num_splits, sum(mse_scores) / num_splits, sum(rmse_scores) / num_splits, sum(r2_scores) / num_splits



	
def	main():
	learningRate = 0.5

	iterations = 1000
	
	mileages, prices = getData(getPath('data.csv'))
	x, y = normalizeData(mileages, prices)
	t0, t1, lossHistory, t0History, t1History = gradientDescent(x, y, learningRate, iterations)
	storeData(t0, t1, 'thetas.csv')
	displayPlot(t0, t1, mileages, prices, lossHistory, t0History, t1History)
	# Dans votre fonction main()
	avg_mae, avg_mse, avg_rmse, avg_r2 = cross_validation(x, y,learningRate, iterations)

	print(f'MAE: \t{avg_mae}')
	print(f'MSE: \t{avg_mse}')
	print(f'RMSE: \t{avg_rmse}')
	print(f'R²: \t{avg_r2}')


	
if	__name__ == '__main__':
	main()
'''