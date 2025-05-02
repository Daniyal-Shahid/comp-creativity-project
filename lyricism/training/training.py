import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load pre-extracted features
df = pd.read_csv('features_30_sec.csv')  # example file name
print(f"Original columns: {df.columns.tolist()}")
print(f"Dataset shape: {df.shape}")

# Remove non-numeric columns (like 'filename')
if 'filename' in df.columns:
    print("Removing 'filename' column which contains string values")
    df = df.drop('filename', axis=1)

# Convert all features to numeric, handling any other non-numeric values
for col in df.columns:
    if col != 'label' and df[col].dtype == 'object':
        print(f"Converting non-numeric column to numeric: {col}")
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop any rows with NaN values (optional)
df = df.dropna()
print(f"Clean dataset shape: {df.shape}")

# Separate features and target
X = df.drop('label', axis=1)  # All columns except the genre label
y = df['label']               # The genre label column

# Step 2: Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 4: Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Print detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=np.unique(y), 
            yticklabels=np.unique(y))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
print("Confusion matrix saved as 'confusion_matrix.png'")

# Save feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

# Save top 15 features to CSV
feature_importance.head(15).to_csv('top_features.csv', index=False)
print("Top 15 features saved to 'top_features.csv'")

# Step 5: Save model
joblib.dump(model, 'genre_predictor.pkl')
print("Model trained and saved!")
