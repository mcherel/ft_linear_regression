import csv

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