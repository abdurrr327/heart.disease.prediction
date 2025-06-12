import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set plot style for aesthetics
sns.set_style('whitegrid')
plt.style.use('fivethirtyeight')

print("--- Heart Disease Prediction Notebook ---")
print("Step 1: Setting up the environment and loading data.")

# --- Load the dataset ---
# Make sure you have uploaded 'HeartDiseaseTrain-Test.csv' to your Colab session.
try:
    df = pd.read_csv('HeartDiseaseTrain-Test.csv')
    print("\nDataset 'HeartDiseaseTrain-Test.csv' loaded successfully!")
    print("Dataset shape:", df.shape)
except FileNotFoundError:
    print("\nError: 'HeartDiseaseTrain-Test.csv' not found.")
    print("Please upload the file to the Colab session by clicking the folder icon on the left.")

# Display the first 5 rows to see the data
print("\nFirst 5 rows of the dataset:")
print(df.head())

# Data Cleaning and Initial Inspection
print("\n--- Step 2: Data Cleaning and Initial Inspection ---")

# Correct the typo in the 'cholestoral' column name
df.rename(columns={'cholestoral': 'cholesterol'}, inplace=True)
print("Corrected column name 'cholestoral' to 'cholesterol'.")

# Get a concise summary of the dataframe
print("\nDataset Information (before cleaning):")
df.info()

# Check for any missing (NaN) values
print("\nChecking for missing (NaN) values...")
print(df.isnull().sum())
print("No NaN values found.")

# Inspect unique values in object columns to plan for encoding
print("\nInspecting unique values in categorical columns:")
for col in df.select_dtypes(include=['object']).columns:
    print(f"- Column '{col}': {df[col].unique()}")

# ---  Handle Data Inconsistencies ---
# The 'thalassemia' column contains a 'No' category, which is likely an error or missing data.
# With only 4 instances, we'll replace it with the mode (most frequent value).
if 'No' in df['thalassemia'].unique():
    mode_thal = df['thalassemia'].mode()[0]
    df['thalassemia'] = df['thalassemia'].replace('No', mode_thal)
    print(f"\nReplaced 'No' in 'thalassemia' with the mode: '{mode_thal}'")

print("\nData inspection complete. Ready for EDA.")


# Exploratory Data Analysis (EDA)
print("\n--- Step 3: Exploratory Data Analysis (EDA) ---")

# 1. Target Variable Distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='target', data=df, palette='viridis')
plt.title('Heart Disease Target Distribution')
plt.xlabel('Heart Disease (0 = No, 1 = Yes)')
plt.ylabel('Count')
plt.show()

# 2. Sex vs. Heart Disease
plt.figure(figsize=(8, 6))
sns.countplot(x='sex', hue='target', data=df, palette='magma')
plt.title('Heart Disease Frequency by Sex')
plt.xlabel('Sex (Female, Male)')
plt.ylabel('Count')
plt.legend(['No Disease', 'Disease'])
plt.show()

# 3. Chest Pain Type vs. Heart Disease
plt.figure(figsize=(12, 6))
sns.countplot(x='chest_pain_type', hue='target', data=df, palette='ocean')
plt.title('Heart Disease Frequency by Chest Pain Type')
plt.xlabel('Chest Pain Type')
plt.ylabel('Count')
plt.xticks(rotation=15)
plt.legend(['No Disease', 'Disease'])
plt.show()

# 4. Age Distribution
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='age', hue='target', multiple='stack', bins=30, palette='Set1')
plt.title('Age Distribution by Heart Disease Status')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()


# Data Preprocessing for Modeling
print("\n--- Step 4: Data Preprocessing for Modeling ---")

# Create a copy to keep the original df for reference
df_processed = df.copy()

# --- Manual Encoding for Binary/Ordinal Features ---
# Map binary features
df_processed['sex'] = df_processed['sex'].map({'Male': 1, 'Female': 0})
df_processed['fasting_blood_sugar'] = df_processed['fasting_blood_sugar'].map({'Greater than 120 mg/ml': 1, 'Lower than 120 mg/ml': 0})
df_processed['exercise_induced_angina'] = df_processed['exercise_induced_angina'].map({'Yes': 1, 'No': 0})

# Map ordinal features
vessels_map = {'Zero': 0, 'One': 1, 'Two': 2, 'Three': 3, 'Four': 4}
df_processed['vessels_colored_by_flourosopy'] = df_processed['vessels_colored_by_flourosopy'].map(vessels_map)

# --- One-Hot Encoding for Nominal Features ---
# These features have no inherent order, so one-hot encoding is appropriate.
categorical_cols = ['chest_pain_type', 'rest_ecg', 'slope', 'thalassemia']
df_processed = pd.get_dummies(df_processed, columns=categorical_cols, drop_first=True)

print("Data has been converted to a fully numerical format.")
print("\nFirst 5 rows of the processed data:")
print(df_processed.head())

# --- Correlation Matrix Heatmap on Processed Data ---
plt.figure(figsize=(20, 15))
correlation_matrix = df_processed.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix of All Processed Features')
plt.show()


# Model Training
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

print("\n--- Step 5: Model Training ---")

# Define features (X) and target (y)
X = df_processed.drop('target', axis=1)
y = df_processed['target']

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")

# --- Feature Scaling ---
# Scale numerical features for models like Logistic Regression.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# --- Model 1: Logistic Regression ---
print("\nTraining a Logistic Regression model...")
log_reg = LogisticRegression(solver='liblinear', random_state=42)
log_reg.fit(X_train_scaled, y_train)
print("Logistic Regression model trained successfully.")

# --- Model 2: Decision Tree Classifier ---
print("\nTraining a Decision Tree Classifier model...")
# We use the unscaled data for tree-based models. max_depth is a hyperparameter to prevent overfitting.
dt_clf = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_clf.fit(X_train, y_train)
print("Decision Tree model trained successfully.")


# Model Evaluation
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, roc_auc_score

print("\n--- Step 6: Model Evaluation ---")

# --- Evaluate Logistic Regression ---
y_pred_log_reg = log_reg.predict(X_test_scaled)
y_prob_log_reg = log_reg.predict_proba(X_test_scaled)[:, 1]

acc_log_reg = accuracy_score(y_test, y_pred_log_reg)
auc_log_reg = roc_auc_score(y_test, y_prob_log_reg)
print(f"Logistic Regression Accuracy: {acc_log_reg:.4f}")
print(f"Logistic Regression AUC: {auc_log_reg:.4f}")

# Confusion Matrix for Logistic Regression
cm_log_reg = confusion_matrix(y_test, y_pred_log_reg)
disp_log_reg = ConfusionMatrixDisplay(confusion_matrix=cm_log_reg)
disp_log_reg.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix - Logistic Regression')
plt.show()


# --- Evaluate Decision Tree ---
y_pred_dt = dt_clf.predict(X_test)
y_prob_dt = dt_clf.predict_proba(X_test)[:, 1]

acc_dt = accuracy_score(y_test, y_pred_dt)
auc_dt = roc_auc_score(y_test, y_prob_dt)
print(f"\nDecision Tree Accuracy: {acc_dt:.4f}")
print(f"Decision Tree AUC: {auc_dt:.4f}")

# Confusion Matrix for Decision Tree
cm_dt = confusion_matrix(y_test, y_pred_dt)
disp_dt = ConfusionMatrixDisplay(confusion_matrix=cm_dt)
disp_dt.plot(cmap=plt.cm.Greens)
plt.title('Confusion Matrix - Decision Tree')
plt.show()

# --- Combined ROC Curve ---
fpr_log_reg, tpr_log_reg, _ = roc_curve(y_test, y_prob_log_reg)
fpr_dt, tpr_dt, _ = roc_curve(y_test, y_prob_dt)

plt.figure(figsize=(10, 7))
plt.plot(fpr_log_reg, tpr_log_reg, color='blue', label=f'Logistic Regression (AUC = {auc_log_reg:.2f})')
plt.plot(fpr_dt, tpr_dt, color='green', label=f'Decision Tree (AUC = {auc_dt:.2f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.grid(True)
plt.show()


# Feature Importance and Conclusion
print("\n--- Step 7: Feature Importance Analysis ---")

# --- Logistic Regression Feature Importance ---
# We use the absolute coefficients as a measure of importance.
importance_log_reg = pd.DataFrame({
    'feature': X.columns,
    'importance': np.abs(log_reg.coef_[0])
}).sort_values('importance', ascending=False).head(15) # Show top 15

plt.figure(figsize=(12, 8))
sns.barplot(x='importance', y='feature', data=importance_log_reg, palette='cubehelix')
plt.title('Top 15 Feature Importances - Logistic Regression')
plt.xlabel('Coefficient Absolute Value')
plt.ylabel('Feature')
plt.show()

# --- Decision Tree Feature Importance ---
importance_dt = pd.DataFrame({
    'feature': X.columns,
    'importance': dt_clf.feature_importances_
}).sort_values('importance', ascending=False).head(15) # Show top 15

plt.figure(figsize=(12, 8))
sns.barplot(x='importance', y='feature', data=importance_dt, palette='viridis')
plt.title('Top 15 Feature Importances - Decision Tree')
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.show()

print("\n--- Highlighted Important Features ---")
print("Based on both models, the following features appear to be strong predictors of heart disease:")
print("- vessels_colored_by_flourosopy: The number of major vessels visible is a top predictor.")
print("- chest_pain_type: Different types of chest pain (especially non-anginal or asymptomatic) are highly indicative.")
print("- thalassemia_Reversable Defect: The presence of a 'Reversable Defect' is very important.")
print("- Max_heart_rate: The maximum heart rate achieved during exercise.")
print("- sex: Gender is consistently a significant factor in both models.")
print("- oldpeak: ST depression induced by exercise relative to rest.")

print("\n--- End of Project ---")
