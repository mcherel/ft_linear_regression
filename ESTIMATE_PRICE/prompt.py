import sys
sys.path.append('../../ft_linear_regression')
import os
from TOOLS import estimate_price as price
import csv
import TOOLS as tools

km = input("Please, enter the milage in kilometers: \n")
try:
    km = int(km)
    if not (0 <= km <= 250000):
        raise ValueError
except ValueError:
    print("Sorry, the asked number should be an int between 0 and 250000")
    exit()

file_path = "../DATA/training.csv"
theta0 = 0
theta1 = 0

if os.path.exists(file_path):
    with open(file=file_path, mode='r', encoding='utf-8') as csvfile:
        dataset = csv.reader(csvfile)
        next(dataset)
        theta0, theta1 = map(float, next(dataset))
        estimated_price = tools.estimate_price.hypothesis(theta0, theta1, km)
print(f"The price of you car is estimated to {int(estimated_price)}")