import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import xgboost as xgb  # Import XGBoost for plotting feature importance

# Step 1: Load the saved model and encoder
with open("xgboost_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("label_encoder.pkl", "rb") as encoder_file:
    label_encoder = pickle.load(encoder_file)

# Step 2: Input a single row of data
# Example of a single row for prediction (change these values based on actual input)
single_row = {
    "profile_completeness": 0.5,
    "post_frequency_per_day": 1,
    "follower_following_ratio": 8.55,
    "average_time_between_posts": 97.11,
    "is_verified": 0
}

# Step 3: Convert single row into a DataFrame for prediction
single_row_df = pd.DataFrame([single_row])

# Step 4: Make the prediction
prediction = model.predict(single_row_df)

# Step 5: Convert the prediction to original label (decode)
predicted_label = label_encoder.inverse_transform(prediction)

# Step 6: Output the prediction
print(f"Predicted behavior label for the input: {predicted_label[0]}")

# Step 7: Load the full dataset for evaluation and visualization
df = pd.read_csv("dataset.csv")
df['behavior_label_encoded'] = label_encoder.transform(df['behavior_label'])

# Features and target
X = df.drop(columns=['account_id', 'behavior_label', 'behavior_label_encoded'])
y = df['behavior_label_encoded']

# Evaluate the model on the full dataset
y_pred = model.predict(X)
accuracy = accuracy_score(y, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
print("\nClassification Report:\n", classification_report(y, y_pred, target_names=label_encoder.classes_))

# Step 8: Visualizations
# 1. Feature Importance
plt.figure(figsize=(8, 6))
xgb.plot_importance(model, max_num_features=10, importance_type='weight')
plt.title("Feature Importance")
plt.show()

# 2. Confusion Matrix
cm = confusion_matrix(y, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
disp.plot(cmap='viridis')
plt.title("Confusion Matrix")
plt.show()

# 3. Class Distribution in Dataset
plt.figure(figsize=(8, 6))
df['behavior_label'].value_counts().plot(kind='bar', color='skyblue')
plt.title("Class Distribution")
plt.xlabel("Behavior Label")
plt.ylabel("Count")
plt.show()

# 4. Correlation Heatmap
corr = df.drop(columns=['account_id', 'behavior_label']).corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()

# 5. Actual vs Predicted Distribution
df['predicted_label'] = label_encoder.inverse_transform(y_pred)
comparison_df = pd.DataFrame({
    'Actual': df['behavior_label'],
    'Predicted': df['predicted_label']
})
plt.figure(figsize=(8, 6))
comparison_df['Predicted'].value_counts().plot(kind='bar', color='green', alpha=0.7, label='Predicted')
comparison_df['Actual'].value_counts().plot(kind='bar', color='orange', alpha=0.7, label='Actual')
plt.title("Actual vs Predicted Distribution")
plt.xlabel("Behavior Label")
plt.ylabel("Count")
plt.legend()
plt.show()

# 6. Distribution of a Feature (e.g., profile_completeness)
plt.figure(figsize=(8, 6))
sns.histplot(df['profile_completeness'], kde=True, bins=10, color='purple')
plt.title("Profile Completeness Distribution")
plt.xlabel("Profile Completeness")
plt.ylabel("Frequency")
plt.show()
