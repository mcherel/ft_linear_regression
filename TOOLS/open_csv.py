import csv

def open_csv(file):
    X = []
    y = []
    with open(file=file, mode='r', encoding='utf-8') as csvfile:
        dataset = csv.reader(csvfile)
        next(dataset)
        for first, second in dataset:
            X.append(float(first))
            y.append(float(second))
    return X, y