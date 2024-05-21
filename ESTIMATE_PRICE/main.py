import sys
sys.path.append('../../ft_linear_regression')
import os
from TOOLS import estimate_price, open_csv

km = input("Please, enter the milage in kilometers: \n")
try:
    km = int(km)
    if not (0 <= km <= 250000):
        raise ValueError
except ValueError:
    print("Sorry, the asked number should be an int between 0 and 250000")
    exit()

training_data = "../DATA/training.csv"

if os.path.exists(training_data):
    theta0, theta1 = open_csv.open_csv(training_data)
    print(f'theta0 : {theta0}')
    print(f'theta1 : {theta1}')
    if not theta0 or not theta1:
        print(f"The training data is not usable")
        exit()
    estimated_price = estimate_price.hypothesis(theta0[0], theta1[0], km)
else:
    print(f"There is no {training_data}")
    exit()
print(f"The price of you car is estimated to {int(estimated_price)}")