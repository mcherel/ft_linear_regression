import sys
sys.path.append('../../ft_linear_regression')
import csv
import random
from TOOLS import open_csv, estimate_price
import numpy as np



def gradient_descent(theta0, theta1, X_km, y_price):
    learning_rate = 0.000001
    num_iterations = 30000
    data_points = len(X_km)


    for i in range(num_iterations):
        #Prediction
        #estimated_prices = price.hypothesis(theta0, theta1, X_km)
        predictions = estimate_price.hypothesis(theta0, theta1, X_km)
        
        # Calculate gradient
        #error = np.mean((predictions - y_price) ** 2)
        # grad_theta0 = (1/data_points) * sum(error[j] - y_price[j] for j in range(data_points))
        # grad_theta1 = (1/data_points) * sum((estimated_prices[j] - y_price[j]) * X_km[j] for j in range(data_points))
        
        # Update params
        error = np.array(predictions) - np.array(y_price)
        theta0 -= learning_rate * (2/data_points) * sum(error[j] - y_price[j] for j in range(data_points))
        theta1 -= learning_rate * (2/data_points) * sum((predictions[j] - y_price[j]) * X_km[j] for j in range(data_points))

        print(f"Iteration {i+1}: theta0={theta0}, theta1={theta1}")

    return theta0, theta1

def main():
    file = "../DATA/data.csv"
    X_km, y_price = open_csv.open_csv(file)


    #Normalizing the data
    X_km_mean = np.mean(X_km)
    X_km_std = np.std(X_km)
    X_km = (X_km - X_km_mean) / X_km_std

    y_price_mean = np.mean(y_price)
    y_price_std = np.std(y_price)
    y_price = (y_price - y_price_mean) / y_price_std
    
    #initializing thetas to random
    #theta0 = random.uniform(-1, 1)
    theta0 = 1
    theta1 = -0.5
    theta0, theta1 = gradient_descent(theta0, theta1, X_km, y_price)
    #theta0, theta1 = gradient_descent(0, 0, X_km, y_price)

    #Denormalizing the data
    theta0 *= y_price_std / X_km_std + y_price_mean - theta1 * (y_price_std * X_km_mean / X_km_std)
    theta1 *= y_price_std / X_km_std

    training_data_file = "../DATA/training.csv"
    with open(training_data_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['theta0', 'theta1'])
        writer.writerow([theta0, theta1])
    print(f'theta0 : {theta0}')
    print(f'theta1 : {theta1}')

if __name__ == '__main__':
    main()
'''
# Initialisation des coefficients
pente = 0
intercept = 0
taux_apprentissage = 0.0001  # Taux d'apprentissage

# Nombre d'itérations
iterations = 1000

# Descente de gradient
for i in range(iterations):
    # Prédiction avec les coefficients actuels
    predictions = pente * annee + intercept
    
    # Calcul de l'erreur
    erreur = np.mean((predictions - prix) ** 2)
    
    # Mise à jour des coefficients avec la descente de gradient
    pente -= taux_apprentissage * (2 / len(annee)) * np.sum((predictions - prix) * annee)
    intercept -= taux_apprentissage * (2 / len(annee)) * np.sum(predictions - prix)
    
# Prédiction du prix pour l'année 2020
prix_predit = pente * 2020 + intercept
print("Prix prédit pour l'année 2020 :", prix_predit)

# Visualisation des données et de la régression linéaire
plt.scatter(annee, prix, color='blue')
plt.plot(annee, pente * annee + intercept, color='red')
plt.xlabel('Année de fabrication')
plt.ylabel('Prix de la voiture')
plt.title('Régression linéaire avec descente de gradient')
plt.show()
'''

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