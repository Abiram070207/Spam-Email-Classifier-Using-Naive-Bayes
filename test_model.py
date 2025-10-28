# test_model.py
# ---------------------------
# This file tests the saved Email Spam Detection model

import pickle
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# ===============================
# 1. Load the saved model/vectorizer
# ===============================
model = pickle.load(open("models/model.pkl", "rb"))
vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))

# ===============================
# 2. Load the same dataset for testing
# (or you can load any unseen emails.csv)
# ===============================
dataset = pd.read_csv('dataset/emails.csv')

# Clean duplicates if any
dataset.drop_duplicates(inplace=True)

# ===============================
# 3. Split the data (same as training split)
# ===============================
from sklearn.model_selection import train_test_split
X = dataset['text']
y = dataset['spam']

# Transform the text using the same vectorizer
X_features = vectorizer.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_features, y, test_size=0.20, random_state=0)

# ===============================
# 4. Make predictions
# ===============================
y_pred = model.predict(X_test)

# ===============================
# 5. Evaluate the model
# ===============================
print("Accuracy:", accuracy_score(y_test, y_pred) * 100, "%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ===============================
# 6. Save test results to file
# ===============================
results = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred
})
results.to_csv("results/test_results.csv", index=False)

print("\nTest results saved to: results/test_results.csv")
