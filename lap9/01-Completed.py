"""
Completed Lap 9 - Part 1: Decision Trees and Random Forests
Dataset: kyphosis.csv
"""

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Get the Data
df = pd.read_csv('kyphosis.csv')
print("=== Data Head ===")
print(df.head())
print("\n=== Data Info ===")
print(df.info())
print("\n=== Data Describe ===")
print(df.describe())

# EDA - Pairplot
print("\nCreating pairplot...")
sns.set_style('whitegrid')
pairplot = sns.pairplot(df, hue='Kyphosis', palette='Set1')
pairplot.savefig('kyphosis_pairplot.png')
print("Pairplot saved to kyphosis_pairplot.png")

# Train Test Split
from sklearn.model_selection import train_test_split
X = df.drop('Kyphosis', axis=1)
y = df['Kyphosis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
print(f"\nTraining set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")

# Decision Trees
from sklearn.tree import DecisionTreeClassifier
print("\n=== Training Decision Tree ===")
dtree = DecisionTreeClassifier(random_state=42)
dtree.fit(X_train, y_train)

# Prediction and Evaluation
predictions = dtree.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix
print("\n=== Decision Tree Evaluation ===")
print("Classification Report:")
print(classification_report(y_test, predictions))
print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))

# Random Forests
from sklearn.ensemble import RandomForestClassifier
print("\n=== Training Random Forest ===")
rfc = RandomForestClassifier(n_estimators=100, random_state=42)
rfc.fit(X_train, y_train)
rfc_pred = rfc.predict(X_test)

print("\n=== Random Forest Evaluation ===")
print("Classification Report:")
print(classification_report(y_test, rfc_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, rfc_pred))

# Compare Models
from sklearn.metrics import accuracy_score
dtree_acc = accuracy_score(y_test, predictions)
rfc_acc = accuracy_score(y_test, rfc_pred)
print("\n=== Model Comparison ===")
print(f"Decision Tree Accuracy: {dtree_acc:.4f}")
print(f"Random Forest Accuracy: {rfc_acc:.4f}")
if rfc_acc > dtree_acc:
    print("Random Forest performed better!")
elif dtree_acc > rfc_acc:
    print("Decision Tree performed better!")
else:
    print("Both models performed equally.")

print("\n=== Lap 9 Part 1 Complete ===")

