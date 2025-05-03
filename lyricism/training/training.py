import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

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

# Step 2: Apply feature scaling (StandardScaler)
print("Applying feature scaling with StandardScaler...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# Step 3: Train/test split using the scaled features
X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y, test_size=0.2, random_state=42)

# Step 4: Train base model
base_model = RandomForestClassifier(n_estimators=100, random_state=42)
base_model.fit(X_train, y_train)

# Step 5: Apply probability calibration
print("Applying probability calibration...")
calibrated_model = CalibratedClassifierCV(base_model, method='sigmoid', cv='prefit')
calibrated_model.fit(X_test, y_test)

# Step 6: Evaluate model
y_pred = calibrated_model.predict(X_test)
y_prob = calibrated_model.predict_proba(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")

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

# Create a calibration plot to visualize how well calibrated the probabilities are
try:
    from sklearn.calibration import calibration_curve
    
    plt.figure(figsize=(10, 8))
    
    # Plot for a few selected classes (if many genres)
    classes = np.unique(y)
    if len(classes) > 5:
        selected_classes = classes[:5]  # Take first 5 classes
    else:
        selected_classes = classes
    
    for i, genre in enumerate(selected_classes):
        y_test_bin = (y_test == genre).astype(int)
        prob_pos = y_prob[:, i]
        
        # Calculate calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_test_bin, prob_pos, n_bins=10)
        
        # Plot calibration curve
        plt.plot(mean_predicted_value, fraction_of_positives, 's-',
                 label=f'{genre}')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
    plt.xlabel('Mean predicted probability')
    plt.ylabel('Fraction of positives')
    plt.title('Calibration Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig('calibration_curve.png')
    print("Calibration curve saved as 'calibration_curve.png'")
except Exception as e:
    print(f"Could not create calibration plot: {str(e)}")

# Save feature importance from the base model
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': base_model.feature_importances_
}).sort_values('importance', ascending=False)

# Save top 15 features to CSV
feature_importance.head(15).to_csv('top_features.csv', index=False)
print("Top 15 features saved to 'top_features.csv'")

# Step 7: Save model and scaler
# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Save the calibrated model
joblib.dump(calibrated_model, 'models/calibrated_genre_predictor.pkl')
print("Calibrated model saved as 'models/calibrated_genre_predictor.pkl'")

# Save the scaler for future use
joblib.dump(scaler, 'models/feature_scaler.pkl')
print("Feature scaler saved as 'models/feature_scaler.pkl'")

# Save the original model too for comparison
joblib.dump(base_model, 'genre_predictor.pkl')
print("Original model saved as 'genre_predictor.pkl'")

print("Training complete!")
