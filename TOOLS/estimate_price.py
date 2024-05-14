import numpy as np

def hypothesis(theta0, theta1, X_km):
    return [theta0 + theta1 * x for x in X_km]
    #return theta1 * np.array(X_km) + theta0



