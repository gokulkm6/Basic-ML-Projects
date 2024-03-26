import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

iris = pd.read_csv("Iris.csv")
print(iris.info())
print(iris.isnull().sum())
print(iris.head())

X = iris.iloc[:,1:5]
Y = iris["Species"]
print(X)
print(Y)

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train,Y_train)

y_pred = model.predict(X_test)
print(y_pred[0])

accuracy = accuracy_score(y_pred,Y_test)
print(accuracy)