import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

bankdata = pd.read_csv("divorce1.csv")

X = bankdata.drop('Class', axis=1)
y = bankdata['Class']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .25)

from sklearn.svm import SVC
svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train, y_train)

y_pred = svclassifier.predict(X_test)

from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))
