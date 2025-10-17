import numpy as np
import matplotlib.pyplot as plt
from data import load_iris_data, load_wine_data
from utils import train_test_split
from knn_classifier import KNNClassifier
from eda import visualize_iris

# Load data
X, y = load_iris_data()

# Optional: visualize
# visualize_iris(X, y, ["sepal length", "sepal width", "petal length", "petal width"])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Evaluate for multiple k values
k_values = [1, 3, 5, 7, 9, 11, 15]
accuracies = []
for k in k_values:
    model = KNNClassifier(k=k)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = np.sum(y_pred == y_test) / len(y_test)
    accuracies.append(acc)
    print(f"k = {k}, Accuracy = {acc*100:.2f}%")

# Plot
plt.figure(figsize=(8, 5))
plt.plot(k_values, accuracies, marker='o')
plt.title("Accuracy vs k-value (Iris Dataset)")
plt.xlabel("k")
plt.ylabel("Accuracy")
plt.grid(True)
plt.show()

# Best k
best_k = k_values[np.argmax(accuracies)]
print(f"\nBest k value: {best_k}")

# Evaluate on Wine dataset
Xw, yw = load_wine_data()
X_train_w, X_test_w, y_train_w, y_test_w = train_test_split(Xw, yw, test_size=0.2, random_state=42)
model_w = KNNClassifier(k=best_k)
model_w.fit(X_train_w, y_train_w)
y_pred_w = model_w.predict(X_test_w)
wine_acc = np.sum(y_pred_w == y_test_w) / len(y_test_w)
print(f"Wine dataset accuracy (k={best_k}): {wine_acc*100:.2f}%")
