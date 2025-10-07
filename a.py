# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv(r"C:\Users\ASUS\Downloads\diabetes_prediction_ - diabetes_prediction_impure.csv.csv")

# Drop rows where target is missing
df_clean = df.dropna(subset=['Diabetic'])

# Fill missing values with median for numerical columns
for col in ['Age', 'BMI', 'Glucose', 'BloodPressure']:
    df_clean[col].fillna(df_clean[col].median(), inplace=True)

# Encode categorical column
df_clean['FamilyHistory'] = df_clean['FamilyHistory'].map({'Yes': 1, 'No': 0})

# Binarize the 'Diabetic' column (>=1 => diabetic)
df_clean['Diabetic'] = (df_clean['Diabetic'] >= 1).astype(int)

print(df_clean.head())

# ========== 📊 EDA Section ==========

# 1. Correlation Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df_clean.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# 2. Count of Diabetic vs Non-Diabetic
sns.countplot(x='Diabetic', data=df_clean, palette='Set2')
plt.title("Count of Diabetic vs Non-Diabetic")
plt.xticks([0, 1], ['No', 'Yes'])
plt.xlabel("Diabetic")
plt.ylabel("Count")
plt.show()

# 3. Boxplots of Features vs Diabetic
features = ['Age', 'BMI', 'Glucose', 'BloodPressure']
for col in features:
    sns.boxplot(x='Diabetic', y=col, data=df_clean)
    plt.title(f'{col} vs Diabetic')
    plt.show()

# 4. Histograms for Feature Distributions
for col in features:
    sns.histplot(df_clean[col], kde=True, color='skyblue')
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.show()

# ========== 🤖 Model Training ==========

# Define features and target
X = df_clean.drop('Diabetic', axis=1)
y = df_clean['Diabetic']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression
log_model = LogisticRegression()
log_model.fit(X_train, y_train)
log_preds = log_model.predict(X_test)

# Train Decision Tree
tree_model = DecisionTreeClassifier(random_state=42)
tree_model.fit(X_train, y_train)
tree_preds = tree_model.predict(X_test)

# ========== ✅ Evaluation ==========

# Evaluation function
def evaluate_model(name, y_true, y_pred):
    print(f"\n{name} Results")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print("Classification Report:\n", classification_report(y_true, y_pred))

# Evaluate models
evaluate_model("Logistic Regression", y_test, log_preds)
evaluate_model("Decision Tree", y_test, tree_preds)

# Confusion Matrix Plots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.heatmap(confusion_matrix(y_test, log_preds), annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title("Logistic Regression")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")

sns.heatmap(confusion_matrix(y_test, tree_preds), annot=True, fmt='d', cmap='Greens', ax=axes[1])
axes[1].set_title("Decision Tree")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Actual")

plt.tight_layout()
plt.show()



# ========== 🧪 Predict for New Input ==========
def get_user_input():
    print("\nEnter patient details:")
    age = float(input("Age: "))
    bmi = float(input("BMI: "))
    glucose = float(input("Glucose: "))
    blood_pressure = float(input("Blood Pressure: "))
    family_history = input("Family History of Diabetes? (Yes/No): ").strip().capitalize()
    family_history_encoded = 1 if family_history == "Yes" else 0

    input_df = pd.DataFrame({
        'Age': [age],
        'BMI': [bmi],
        'Glucose': [glucose],
        'BloodPressure': [blood_pressure],
        'FamilyHistory': [family_history_encoded]
    })

    return input_df

# Get input from user
user_input_df = get_user_input()

# Logistic Regression Prediction
log_result = log_model.predict(user_input_df)[0]
print("\n🧠 Logistic Regression Prediction:", "Diabetic" if log_result == 1 else "Not Diabetic")

# Decision Tree Prediction
tree_result = tree_model.predict(user_input_df)[0]
print("🌳 Decision Tree Prediction:", "Diabetic" if tree_result == 1 else "Not Diabetic")