#---------------------------------------------------------------------------
# Author:       Jordan Krause
# Class:        SWE-452
#----------------------------------------------------------------------------

from sklearn.datasets import load_iris
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

iris = load_iris()

# Features (input data)
X = iris.data

# Target (output data)
y = iris.target

feature_names = iris.feature_names
target_names = iris.target_names

print("Feature names:", feature_names)
print("Target names:", target_names)    
print("\nFirst 5 Samples:\n", X[:5])

data = pd.DataFrame(iris.data, columns=feature_names)

print("\nMissing Values per feature:", data.isnull().sum())

print("Number of duplicate rows:", data.duplicated().sum())

data.drop_duplicates(inplace=True)

print("\nMissing Values per feature:", data.isnull().sum())

print("Number of duplicate rows:", data.duplicated().sum())

#! Data Preprocessing -> Standardization of Features
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("\nFirst 5 rows of scaled training data:\n", X_train[:5]) 

# Train and evaluate the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy*100:.2f}%")

# Generate a classification report
print("/nClassification Report:\n", classification_report(y_test, y_pred, target_names=target_names))

# Generate a confusion matrix
cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()  