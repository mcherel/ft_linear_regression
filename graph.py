# https://www.kaggle.com/datasets/devchauhan1/salary-datacsv -> Salary_Data.csv

import csv
import turtle

# Graph Window Initialisation
window = turtle.Screen()
window.title("Salary")

# Turtle creation
scatter = turtle.Turtle()
scatter.speed(0) # Maximal speed - no animation


X = []
y = []
with open("Salary_Data.csv", mode='r', encoding='utf-8') as csvfile:
    dataset = csv.reader(csvfile)
    next(dataset) # Ignoring Title
    for data in dataset:
        X.append(float(data[0]))
        y.append(float(data[1]))

scale = 400
dot_size = 7
margin = 50

# Scaling
max_x = max(X)
min_x = min(X)
max_y = max(y)
min_y = min(y)
scale_x = scale / max_x # x values scale
scale_y = scale / max_y # y values scale



# Draw the scatter plot
for i in range(len(X)):
    scatter.penup()
    scatter.goto(X[i] * scale_x - scale, y[i] * scale_y - scale)  # Scale and shift to fit in window
    scatter.dot(dot_size+2,'blue')
    scatter.dot(dot_size,'cyan')
    scatter.pen(shown=0) # hiding the runner

    # Drawing the rectangle
    scatter.penup()
    scatter.goto(min_x * scale_x - scale - margin, min_y * scale_y - scale - margin)
    scatter.pendown()
    scatter.goto(max_x * scale_x - scale + margin*2, min_y * scale_y - scale - margin)
    scatter.goto(max_x * scale_x - scale  + margin*2, max_y * scale_y - scale  + margin)
    scatter.goto(min_x * scale_x - scale  - margin, max_y * scale_y - scale  + margin)
    scatter.goto(min_x * scale_x - scale  - margin, min_y * scale_y - scale  - margin)

# Dessiner les graduations sur l'axe x
for i in range(1, int(max_x) + 3):
    x_tick = i * scale//max_x + 1 - scale
    scatter.goto(x_tick, min_y * scale_y - scale - margin)  # Déplacer vers le bas pour l'axe x
    scatter.pendown()
    scatter.goto(x_tick, min_y * scale_y - scale - margin - 5)  # Dessiner la ligne
    scatter.penup()
    scatter.goto(x_tick - 5, min_y * scale_y - scale - margin - 15)  # Positionnement de l'étiquette
    scatter.write(str(i), align="center")

# Dessiner les graduations sur l'axe y
for i in range(round(int(min_y), -4), round(int(max_y), -4) + 3, 20000):
    y_tick = i * scale_y - scale
    scatter.penup()
    scatter.goto(min_x * scale_x - scale - margin, y_tick)  # Déplacer vers la gauche pour l'axe y
    scatter.pendown()
    scatter.goto(min_x * scale_x - scale - margin - 5, y_tick)  # Dessiner la ligne
    scatter.penup()
    scatter.goto(min_x * scale_x - scale - margin - 25, y_tick - 10)  # Positionnement de l'étiquette
    scatter.write(str(i), align="right")
# Whait user to close the window
# **Metrics**
# Mean Absolute Error / MAE
# Mean Squared Error / MSE
# Root Mean Squared Error / RMSE
# R Square R2
turtle.done()

