import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.feature_selection import SelectKBest, f_classif
import joblib

data_dir = 'data'
models_dir = 'models'
reports_dir = 'reports'

os.makedirs(data_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)
os.makedirs(reports_dir, exist_ok=True)

data_path = os.path.join(data_dir, '/content/drive/MyDrive/IRIS (1).csv')
try:
    data = pd.read_csv(data_path)
except FileNotFoundError:
    raise FileNotFoundError(f"Dataset file not found at {data_path}. Please place your dataset in the '{data_dir}' folder.")

print("Dataset Preview:")
print(data.head())

print("\nSummary Statistics:")
print(data.describe())

print("\nMissing Values:")
print(data.isnull().sum())

sns.pairplot(data, hue='species')
plt.show()

correlation_matrix = data.corr(numeric_only=True)
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.show()

X = data.drop('species', axis=1)
y = data['species']

target_names = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
if y.dtype == 'object':
    pass
else:
    y = y.map(target_names)

selector = SelectKBest(score_func=f_classif, k=2)
X_selected = selector.fit_transform(X, y)
selected_features = X.columns[selector.get_support()]
print("\nSelected Features:", selected_features)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

logreg = LogisticRegression(random_state=42, max_iter=200)
logreg.fit(X_train_scaled, y_train)
y_pred_logreg = logreg.predict(X_test_scaled)
accuracy_logreg = accuracy_score(y_test, y_pred_logreg)
print(f"\nLogistic Regression Accuracy: {accuracy_logreg:.2f}")
print(classification_report(y_test, y_pred_logreg))

svm = SVC(kernel='linear', random_state=42)
svm.fit(X_train_scaled, y_train)
y_pred_svm = svm.predict(X_test_scaled)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print(f"SVM Accuracy: {accuracy_svm:.2f}")
print(classification_report(y_test, y_pred_svm))

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)
y_pred_rf = rf.predict(X_test_scaled)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Accuracy: {accuracy_rf:.2f}")
print(classification_report(y_test, y_pred_rf))


models = {
    "Logistic Regression": accuracy_logreg,
    "SVM": accuracy_svm,
    "Random Forest": accuracy_rf
}
print("\nModel Accuracies:")
for model, acc in models.items():
    print(f"{model}: {acc:.2f}")

conf_matrix = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_names.values(), yticklabels=target_names.values())
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
confusion_matrix_path = os.path.join(reports_dir, 'confusion_matrix.png')
plt.savefig(confusion_matrix_path)
plt.show()
print(f"Confusion matrix saved to {confusion_matrix_path}")

model_path = os.path.join(models_dir, 'iris_classifier.pkl')
joblib.dump(rf, model_path)
print(f"\nModel saved to {model_path}")
