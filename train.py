import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import xgboost as xgb
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the dataset
df = pd.read_csv("dataset.csv")

# Step 2: Preprocess the data
# Encode the target variable
label_encoder = LabelEncoder()
df['behavior_label_encoded'] = label_encoder.fit_transform(df['behavior_label'])

# Features and target
X = df.drop(columns=['account_id', 'behavior_label', 'behavior_label_encoded'])
y = df['behavior_label_encoded']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train the XGBoost model
model = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    objective="multi:softmax",
    num_class=len(label_encoder.classes_),
    random_state=42
)

model.fit(X_train, y_train)

# Step 4: Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Save the model and encoder
with open("xgboost_model2.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

with open("label_encoder2.pkl", "wb") as encoder_file:
    pickle.dump(label_encoder, encoder_file)

print("Model and encoder saved successfully!")

# Step 5: Visualizations

# 1. Feature Importance
xgb.plot_importance(model, max_num_features=10, importance_type='weight')
plt.title("Feature Importance")
plt.show()

# 2. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
disp.plot(cmap='viridis')
plt.title("Confusion Matrix")
plt.show()

# 3. Class Distribution in Dataset
df['behavior_label'].value_counts().plot(kind='bar', color='skyblue')
plt.title("Class Distribution")
plt.xlabel("Behavior Label")
plt.ylabel("Count")
plt.show()

# 5. Correlation Heatmap
corr = df.drop(columns=['account_id', 'behavior_label']).corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()

# 6. Prediction Distribution
# Add predictions to the test set for comparison
y_test_labels = label_encoder.inverse_transform(y_test)
y_pred_labels = label_encoder.inverse_transform(y_pred)

# Create a DataFrame for comparison
comparison_df = pd.DataFrame({'Actual': y_test_labels, 'Predicted': y_pred_labels})

# Plot the comparison
comparison_df['Predicted'].value_counts().plot(kind='bar', color='lightgreen', alpha=0.7, label='Predicted')
comparison_df['Actual'].value_counts().plot(kind='bar', color='orange', alpha=0.7, label='Actual')
plt.title("Prediction vs Actual Distribution")
plt.xlabel("Behavior Label")
plt.ylabel("Count")
plt.legend()
plt.show()
