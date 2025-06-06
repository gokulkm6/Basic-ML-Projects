import numpy as np
import pandas as pd

data=pd.read_csv("fraudTrain.csv")

data.drop(columns=['Unnamed: 0','cc_num','first', 'last', 'street', 'city', 'state', 'zip', 'dob', 'trans_num'],inplace=True)

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
newd = data.apply(LabelEncoder().fit_transform)

from sklearn.model_selection import train_test_split
X = newd.drop("is_fraud", axis=1)
y = newd["is_fraud"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)

y_pred = dt.predict(X_test)

print("Accuracy Score: ", accuracy_score(y_test, y_pred))

fraud_prob = dt.predict_proba(X_test)[0][1]
print("Fruad Chance:", fraud_prob)