import sys
sys.path.append('../../ft_linear_regression')
import os
from TOOLS import estimate_price as price
import csv
import TOOLS as tools

km = int(input("Please, enter the milage in kilometers: \n"))

file_path = "../DATA/training.csv"
theta0 = 0
theta1 = 0

if os.path.exists(file_path):
    with open(file=file_path, mode='r', encoding='utf-8') as csvfile:
        dataset = csv.reader(csvfile)
        next(dataset)
        theta0, theta1 = map(float, next(dataset))
print(f"The price of you car is estimated to {int(tools.estimate_price.hypothesis(theta0, theta1, [km])[0])}")