Here is the code behind my CS 1111/204 project:

```python
from IPython import get_ipython
from IPython.display import display
from google.colab import drive
drive.mount('/content/drive')
filepath = '/content/drive/MyDrive/heart_data.csv'
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib as mpl
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn.metrics import r2_score
from sklearn.linear_model import LogisticRegression



data = pd.read_csv(filepath)

data['age'] = data['age'].astype(int)
data['platelets'] = data['platelets'].astype(int)
data.describe().round(2)```
```python
data.hist(figsize=(15,12))
plt.suptitle('heart failure dataset data histogram' )
plt.show()
```
```python
x = data.drop('DEATH_EVENT', axis=1)
x = x.drop('time', axis=1)
y = data['DEATH_EVENT']

confusion_matrices = {}
scores = {}
testscores = {}

columns_to_drop = x.columns

highest_fnr = 0
highest_fnr_column = None

column_accuracies = {}

for column in columns_to_drop:
    X_modified = x.drop(columns=[column])

    kf = KFold(n_splits=5, shuffle=True, random_state=0)

    cms = []  

    train_scores = []
    test_scores = []
    
    for train_index, test_index in kf.split(X_modified):
        X_train, X_test = X_modified.iloc[train_index], X_modified.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        regr = LogisticRegression(max_iter=20000)
        regr.fit(X_train, y_train)

        train_score = regr.score(X_train, y_train)
        test_score = regr.score(X_test, y_test)
        train_scores.append(train_score)
        test_scores.append(test_score)

        y_pred = regr.predict(X_test)
        y_pred = np.where(y_pred >= 0.5, 1, 0)

    avg_train_score = np.mean(train_scores)
    avg_test_score = np.mean(test_scores)
    print(f"Average Train Score when dropping {column}: {avg_train_score:.2f}")
    print(f"Average Test Score when dropping {column}: {avg_test_score:.2f}")
    print()

    cm = confusion_matrix(y_test, y_pred)
    cms.append(cm)

    avg_cm = np.mean(cms, axis=0)

    FN = avg_cm[1, 0]
    TP = avg_cm[1, 1]
    FNR = FN / (FN + TP) if (FN + TP) != 0 else 0

    if FNR > highest_fnr:
        highest_fnr = FNR
        highest_fnr_column = column
    
    TN = avg_cm[0, 0]
    FP = avg_cm[0, 1]
    FN = avg_cm[1, 0]
    TP = avg_cm[1, 1]

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0  
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0  
    f1_score = 2 * (precision * recall)
    specificity = TN / (TN + FP) if (TN + FP) != 0 else 0

    disp = ConfusionMatrixDisplay(confusion_matrix=avg_cm)
    disp.plot(cmap='Blues', values_format='.0f')
    plt.xlabel(f"Predicted Label\nAccuracy: {accuracy:.2f}, Precision: {precision:.2f}\nRecall: {recall:.2f}, F1-score: {f1_score:.2f}\nSpecificity: {specificity:.2f}")
    plt.ylabel("True Label")
    plt.title(f"Average Confusion Matrix - Dropped Column: {column}")
    plt.text(0, 0.8, 'False Neg', ha='center', va='center', color='red')
    plt.text(1, 0.8, 'True Pos', ha='center', va='center', color='white')
    plt.text(0, -0.2, 'True Neg', ha='center', va='center', color='white')
    plt.text(1, -0.2, 'False Pos', ha='center', va='center', color='blue')
    plt.show()
    print()
    print()

    column_accuracies[column] = accuracy

    testscores[column] = avg_test_score.round(3)

    confusion_matrices[column] = avg_cm
    scores[column] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'specificity': specificity
    }

ranked_columns = sorted(testscores.items(), key=lambda item: item[1], reverse=True)
print()

for column, score in ranked_columns:
    print(f"Column: {column}, Test R^2 score: {score}")

## plot a histogram with the column names on the x axis and the scores from testscores on the y axis
column_names = list(testscores.keys())
scores = list(testscores.values())

plt.bar(column_names, scores)
plt.ylabel("R^2 Scores")
plt.title("Feature Ranking")
plt.xticks(rotation=90)  
plt.tight_layout()  
plt.ylim(0.6, 0.8)
plt.show()
```
```python
x = data.drop('DEATH_EVENT', axis=1)
x = x.drop('time', axis=1)
y = data['DEATH_EVENT']

columns_to_drop = x.columns

kf = KFold(n_splits=5, shuffle=True, random_state=0)

accuracies = []
recalls = []
precisions = []
f1_scores = []
specificities = []

def evaluate_and_plot(X, y, column_dropped=None):
    cms = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        regr = LogisticRegression(max_iter=10000)
        regr.fit(X_train, y_train)

        y_pred = regr.predict(X_test)
        y_pred = np.where(y_pred >= 0.5, 1, 0)

        cm = confusion_matrix(y_test, y_pred)
        cms.append(cm)

    avg_cm = np.mean(cms, axis=0)

    TN, FP, FN, TP = avg_cm.ravel()
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    accuracies.append(accuracy)
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    precisions.append(precision)
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    recalls.append(recall)
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    f1_scores.append(f1_score)
    specificity = TN / (TN + FP) if (TN + FP) != 0 else 0
    specificities.append(specificity)

num_to_drop_start = 1
num_to_drop = 1

for num_to_drop in range(num_to_drop_start, len(columns_to_drop)):
  columns_to_drop_final = [col for col, _ in ranked_columns[:num_to_drop]]

  X_final = x.drop(columns=columns_to_drop_final)
  evaluate_and_plot(X_final, y, column_dropped=columns_to_drop_final)



fig, axes = plt.subplots(1, 5, figsize=(12, 5))  

x_values = range(num_to_drop_start, len(columns_to_drop))
max_accuracy = max(accuracies)
x_max = accuracies.index(max_accuracy) + num_to_drop_start
axes[0].plot(x_values, accuracies)
axes[0].plot(x_max, max_accuracy, 'ro')  
axes[0].set_xlabel("Number of Dropped Columns")
axes[0].set_ylabel("Accuracy")
axes[0].annotate(f"({x_max}, {max_accuracy:.2f})",
                 xy=(x_max, max_accuracy),
                 xytext=(x_max + 0.5, max_accuracy + 0.01),
                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))

max_recall = max(recalls)
x_max = recalls.index(max_recall) + num_to_drop_start
axes[1].plot(x_values, recalls)
axes[1].plot(x_max, max_recall, 'ro')  
axes[1].set_xlabel("Number of Dropped Columns")
axes[1].set_ylabel("Recall")
axes[1].annotate(f"({x_max}, {max_recall:.2f})",
                 xy=(x_max, max_recall),
                 xytext=(x_max + 0.5, max_recall + 0.01),
                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))

max_precision = max(precisions)
x_max = precisions.index(max_precision) + num_to_drop_start
axes[2].plot(x_values, precisions)
axes[2].plot(x_max, max_precision, 'ro')
axes[2].set_xlabel("Number of Dropped Columns")
axes[2].set_ylabel("Precision")
axes[2].annotate(f"({x_max}, {max_precision:.2f})",
                 xy=(x_max, max_precision),
                 xytext=(x_max + 0.5, max_precision + 0.0),
                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))

max_f1 = max(f1_scores)
x_max = f1_scores.index(max_f1) + num_to_drop_start
axes[3].plot(x_values, f1_scores)
axes[3].plot(x_max, max_f1, 'ro')  
axes[3].set_xlabel("Number of Dropped Columns")
axes[3].set_ylabel("F1 score")
axes[3].annotate(f"({x_max}, {max_f1:.2f})",
                 xy=(x_max, max_f1),
                 xytext=(x_max + 0.5, max_f1 + 0.01),
                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))

max_specificity = max(specificities)
x_max = specificities.index(max_specificity) + num_to_drop_start
axes[4].plot(x_values, specificities)
axes[4].plot(x_max, max_specificity, 'ro')  
axes[4].set_xlabel("Number of Dropped Columns")
axes[4].set_ylabel("Specificity")
axes[4].annotate(f"({x_max}, {max_specificity:.2f})",
                 xy=(x_max, max_specificity),
                 xytext=(x_max + 0.5, max_specificity + 0.01),
                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))



plt.tight_layout()

plt.show()

remaining_columns = X_final.columns  
print("Remaining Columns:", remaining_columns)
```
```python
x = data[['serum_creatinine','ejection_fraction']]  
y = data['DEATH_EVENT']

kf = KFold(n_splits=5, shuffle=True, random_state=0)

train_scores = []
test_scores = []
cms = []

for train_index, test_index in kf.split(x):
    X_train, X_test = x.iloc[train_index], x.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    regr = LogisticRegression(max_iter=10000)
    regr.fit(X_train, y_train)

    train_score = regr.score(X_train, y_train)
    test_score = regr.score(X_test, y_test)
    train_scores.append(train_score)
    test_scores.append(test_score)

    y_pred = regr.predict(X_test)
    y_pred = np.where(y_pred >= 0.5, 1, 0)
    
    cm = confusion_matrix(y_test, y_pred)
    cms.append(cm)

avg_train_score = np.mean(train_scores)
avg_test_score = np.mean(test_scores)
avg_cm = np.mean(cms, axis=0)

print(f"Average Train Score: {avg_train_score:.2f}")
print(f"Average Test Score: {avg_test_score:.2f}")

TN, FP, FN, TP = avg_cm.ravel()
accuracy = (TP + TN) / (TP + TN + FP + FN)
precision = TP / (TP + FP) if (TP + FP) != 0 else 0
recall = TP / (TP + FN) if (TP + FN) != 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
specificity = TN / (TN + FP) if (TN + FP) != 0 else 0

disp = ConfusionMatrixDisplay(confusion_matrix=avg_cm)
disp.plot(cmap='Blues', values_format='.0f')
plt.xlabel(f"Predicted Label\nAccuracy: {accuracy:.2f}, Precision: {precision:.2f}\nRecall: {recall:.2f}, F1-score: {f1_score:.2f}\nSpecificity: {specificity:.2f}")
plt.ylabel("True Label")
plt.title(f"Average Confusion Matrix with interest features only")
plt.text(0, 0.8, 'False Neg', ha='center', va='center', color='red')
plt.text(1, 0.8, 'True Pos', ha='center', va='center', color='white')
plt.text(0, -0.2, 'True Neg', ha='center', va='center', color='white')
plt.text(1, -0.2, 'False Pos', ha='center', va='center', color='blue')
plt.show()
print()
print()
```
```python
serum_creatinine = float(input("Enter serum creatinine value: "))
ejection_fraction = float(input("Enter ejection fraction: "))

user_input = pd.DataFrame([[serum_creatinine, ejection_fraction]],
                           columns=['serum_creatinine', 'ejection_fraction'])  

# Make prediction
prediction = regr.predict(user_input)
predicted_death_event = np.where(prediction >= 0.5, 1, 0)[0]

print(f"Predicted Death Event: {predicted_death_event}")
```
```python
from sklearn.metrics import RocCurveDisplay

roc_display = RocCurveDisplay.from_estimator(regr,X_test,y_test)
plt.plot([0,1],[0,1],linestyle='--')
plt.show
```
```python
x = data.drop('DEATH_EVENT', axis=1)
x = x.drop('time', axis=1)
y = data['DEATH_EVENT']

kf = KFold(n_splits=5, shuffle=True, random_state=0)

train_scores = []
test_scores = []
cms = []

for train_index, test_index in kf.split(x):
    X_train, X_test = x.iloc[train_index], x.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    regr = LogisticRegression(max_iter=10000)
    regr.fit(X_train, y_train)

    train_score = regr.score(X_train, y_train)
    test_score = regr.score(X_test, y_test)
    train_scores.append(train_score)
    test_scores.append(test_score)

    y_pred = regr.predict(X_test)
    y_pred = np.where(y_pred >= 0.5, 1, 0)
    
    cm = confusion_matrix(y_test, y_pred)
    cms.append(cm)

avg_train_score = np.mean(train_scores)
avg_test_score = np.mean(test_scores)
avg_cm = np.mean(cms, axis=0)

print(f"Average Train Score: {avg_train_score:.2f}")
print(f"Average Test Score: {avg_test_score:.2f}")

TN, FP, FN, TP = avg_cm.ravel()
accuracy = (TP + TN) / (TP + TN + FP + FN)
precision = TP / (TP + FP) if (TP + FP) != 0 else 0
recall = TP / (TP + FN) if (TP + FN) != 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
specificity = TN / (TN + FP) if (TN + FP) != 0 else 0

disp = ConfusionMatrixDisplay(confusion_matrix=avg_cm)
disp.plot(cmap='Blues', values_format='.0f')
plt.xlabel(f"Predicted Label\nAccuracy: {accuracy:.2f}, Precision: {precision:.2f}\nRecall: {recall:.2f}, F1-score: {f1_score:.2f}\nSpecificity: {specificity:.2f}")
plt.ylabel("True Label")
plt.title(f"Average Confusion Matrix - full model")
plt.text(0, 0.8, 'False Neg', ha='center', va='center', color='red')
plt.text(1, 0.8, 'True Pos', ha='center', va='center', color='white')
plt.text(0, -0.2, 'True Neg', ha='center', va='center', color='white')
plt.text(1, -0.2, 'False Pos', ha='center', va='center', color='blue')
plt.show()
print()
print()

roc_display = RocCurveDisplay.from_estimator(regr,X_test,y_test)
plt.plot([0,1],[0,1],linestyle='--')
plt.show
```

[back](./)
