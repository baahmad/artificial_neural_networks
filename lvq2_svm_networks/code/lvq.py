import numpy as np
from neupy import algorithms
import pandas as pd

bankdata = pd.read_csv("divorce2.csv")
bankdata2 = pd.read_csv("divorce3.csv")

X = bankdata.drop('Class', axis=1)
y = bankdata['Class']

lvqnet = algorithms.LVQ21(n_inputs=54, n_classes=2)
lvqnet.train(X, y, epochs=100)

X_test = bankdata2.drop('Class', axis=1)
y_test = bankdata2['Class']

predictions = lvqnet.predict(X_test)

Num0 = 0
Num1 = 0
Correct0 = 0
Correct1 = 0

for x in range(len(y_test)):
    if y_test[x] == 1:
        Num1 += 1
    elif y_test[x] == 0:
        Num0 += 1
    if predictions[x] == 1 and y_test[x] == 1:
        Correct1 += 1
    elif predictions[x] == 0 and y_test[x] == 0:
        Correct0 += 1

ratio0 = Correct0/Num0
ratio1 = Correct1/Num1

print([ratio0, ratio1])
