# Helper script to validate Lap 10 assignment logic for Iris SVM.
# Not required by the notebook, but used to ensure code correctness.

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV

# Load iris
iris = sns.load_dataset('iris')

# Pairplot
_ = sns.pairplot(iris, hue='species')
plt.close('all')

# KDE plot for setosa
setosa = iris[iris['species'] == 'setosa']
plt.figure()
_ = sns.kdeplot(data=setosa, x='sepal_length', y='sepal_width', fill=True, cmap='viridis')
plt.close('all')

X = iris.drop(columns=['species'])
y = iris['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

model = SVC()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))

param_grid = {
    'C': [0.1, 1, 10],
    'gamma': [0.1, 0.01, 0.001],
    'kernel': ['rbf']
}

grid = GridSearchCV(SVC(), param_grid=param_grid, cv=3, verbose=0)
grid.fit(X_train, y_train)

grid_predictions = grid.predict(X_test)
print('Best params:', grid.best_params_)
print(confusion_matrix(y_test, grid_predictions))
print(classification_report(y_test, grid_predictions))

