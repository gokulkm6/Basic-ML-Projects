import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

data=pd.read_csv("Churn_Modelling.csv")

X = data[["RowNumber","CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "HasCrCard","IsActiveMember","EstimatedSalary"]]
y = data["Exited"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

Y_pred = rf_model.predict(X_test)

accuracy = accuracy_score(y_test, Y_pred)
print("Accuracy:", accuracy)

confusion_matrix = metrics.confusion_matrix(y_test, Y_pred)

x=metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
x.plot()
plt.show()

new_customer = pd.DataFrame(X)

new_customer = scaler.transform(new_customer)

churn_prob = rf_model.predict_proba(new_customer)[0][1]
print("Churn Probability:", churn_prob)