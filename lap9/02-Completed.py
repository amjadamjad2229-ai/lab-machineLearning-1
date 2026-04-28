"""
Completed Lap 9 - Part 2: Decision Trees and Random Forest Project
Dataset: loan_data.csv
"""

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Get the Data
loans = pd.read_csv('loan_data.csv')
print("=== Loans Info ===")
print(loans.info())
print("\n=== Loans Head ===")
print(loans.head())
print("\n=== Loans Describe ===")
print(loans.describe())

# Exploratory Data Analysis
print("\nCreating EDA visualizations...")

# 1. Histogram of two FICO distributions on top of each other, one for each credit.policy outcome
plt.figure(figsize=(10, 6))
loans[loans['credit.policy'] == 1]['fico'].hist(alpha=0.5, bins=30, label='Credit Policy = 1', color='blue')
loans[loans['credit.policy'] == 0]['fico'].hist(alpha=0.5, bins=30, label='Credit Policy = 0', color='red')
plt.legend()
plt.xlabel('FICO Score')
plt.title('FICO Distribution by Credit Policy')
plt.savefig('fico_by_credit_policy.png')
print("Saved: fico_by_credit_policy.png")

# 2. Similar figure, except this time select by the not.fully.paid column
plt.figure(figsize=(10, 6))
loans[loans['not.fully.paid'] == 1]['fico'].hist(alpha=0.5, bins=30, label='Not Fully Paid = 1', color='red')
loans[loans['not.fully.paid'] == 0]['fico'].hist(alpha=0.5, bins=30, label='Not Fully Paid = 0', color='blue')
plt.legend()
plt.xlabel('FICO Score')
plt.title('FICO Distribution by Not Fully Paid')
plt.savefig('fico_by_not_fully_paid.png')
print("Saved: fico_by_not_fully_paid.png")

# 3. Countplot using seaborn showing the counts of loans by purpose, with the color hue defined by not.fully.paid
plt.figure(figsize=(12, 6))
sns.countplot(data=loans, x='purpose', hue='not.fully.paid', palette='Set1')
plt.title('Loan Purpose Count by Not Fully Paid')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('purpose_countplot.png')
print("Saved: purpose_countplot.png")

# 4. Jointplot between FICO score and interest rate
jointplot = sns.jointplot(data=loans, x='fico', y='int.rate', color='purple')
jointplot.fig.suptitle('FICO Score vs Interest Rate', y=1.02)
jointplot.savefig('fico_int_rate_jointplot.png')
print("Saved: fico_int_rate_jointplot.png")

# 5. Lmplots to see if the trend differed between not.fully.paid and credit.policy
lmplot = sns.lmplot(data=loans, x='fico', y='int.rate', hue='credit.policy', col='not.fully.paid', palette='Set1')
lmplot.fig.suptitle('FICO vs Interest Rate by Credit Policy and Not Fully Paid', y=1.02)
lmplot.savefig('fico_int_rate_lmplot.png')
print("Saved: fico_int_rate_lmplot.png")

# Setting up the Data
print("\n=== Setting up Data ===")
print("Loans info again:")
print(loans.info())

# Categorical Features
# Create a list of 1 element containing the string 'purpose'. Call this list cat_feats.
cat_feats = ['purpose']
print(f"\nCategorical features: {cat_feats}")

# Use pd.get_dummies to create a fixed larger dataframe with dummy variables
final_data = pd.get_dummies(loans, columns=cat_feats, drop_first=True)
print(f"\nFinal data shape after get_dummies: {final_data.shape}")
print(f"Final data columns: {list(final_data.columns)}")

# Train Test Split
from sklearn.model_selection import train_test_split
X = final_data.drop('not.fully.paid', axis=1)
y = final_data['not.fully.paid']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
print(f"\nTraining set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")

# Training a Decision Tree Model
from sklearn.tree import DecisionTreeClassifier
print("\n=== Training Decision Tree ===")
dtree = DecisionTreeClassifier(random_state=42)
dtree.fit(X_train, y_train)

# Predictions and Evaluation of Decision Tree
from sklearn.metrics import classification_report, confusion_matrix
dtree_pred = dtree.predict(X_test)
print("\n=== Decision Tree Evaluation ===")
print("Classification Report:")
print(classification_report(y_test, dtree_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, dtree_pred))

# Training the Random Forest model
from sklearn.ensemble import RandomForestClassifier
print("\n=== Training Random Forest ===")
rfc = RandomForestClassifier(n_estimators=100, random_state=42)
rfc.fit(X_train, y_train)

# Predictions and Evaluation
rfc_pred = rfc.predict(X_test)
print("\n=== Random Forest Evaluation ===")
print("Classification Report:")
print(classification_report(y_test, rfc_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, rfc_pred))

# What performed better the random forest or the decision tree?
from sklearn.metrics import accuracy_score
dtree_acc = accuracy_score(y_test, dtree_pred)
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

print("\n=== Lap 9 Part 2 Complete ===")

