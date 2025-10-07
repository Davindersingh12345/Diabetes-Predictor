# ========== 📦 Import Libraries ==========
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# ========== 📥 Load Dataset ==========
df = pd.read_csv(r"C:\Users\ASUS\Downloads\diabetes_prediction_ - diabetes_prediction_impure.csv.csv")  # Adjust path if needed

# ========== 🧹 Data Cleaning ==========
df_clean = df.dropna(subset=['Diabetic']).copy()

# Fill missing values in numeric columns with median
for col in ['Age', 'BMI', 'Glucose', 'BloodPressure']:
    df_clean[col].fillna(df_clean[col].median(), inplace=True)

# Encode FamilyHistory (Yes=1, No=0)
df_clean['FamilyHistory'] = df_clean['FamilyHistory'].map({'Yes': 1, 'No': 0})

# Binarize Diabetic column
df_clean['Diabetic'] = (df_clean['Diabetic'] >= 1).astype(int)

# ========== 📊 EDA ==========
print("\n🔍 First 5 Cleaned Rows:")
print(df_clean.head())

print("\n📈 Correlation Heatmap:")
plt.figure(figsize=(8, 6))
sns.heatmap(df_clean.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

print("\n📊 Count of Diabetic vs Non-Diabetic:")
sns.countplot(x='Diabetic', data=df_clean, palette='Set2')
plt.title("Diabetic Distribution")
plt.xticks([0, 1], ['No', 'Yes'])
plt.xlabel("Diabetic")
plt.ylabel("Count")
plt.show()

# Boxplots
for col in ['Age', 'BMI', 'Glucose', 'BloodPressure']:
    sns.boxplot(x='Diabetic', y=col, data=df_clean)
    plt.title(f'{col} vs Diabetic')
    plt.show()

# Histograms
for col in ['Age', 'BMI', 'Glucose', 'BloodPressure']:
    sns.histplot(df_clean[col], kde=True, color='skyblue')
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.show()

# ========== 🧠 Model Training ==========
X = df_clean.drop('Diabetic', axis=1)
y = df_clean['Diabetic']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression
log_model = LogisticRegression()
log_model.fit(X_train, y_train)
log_preds = log_model.predict(X_test)

# Decision Tree
tree_model = DecisionTreeClassifier(random_state=42)
tree_model.fit(X_train, y_train)
tree_preds = tree_model.predict(X_test)

# ========== 📊 Evaluation ==========
def evaluate_model(name, y_true, y_pred):
    print(f"\n📋 {name} Evaluation:")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print("Classification Report:\n", classification_report(y_true, y_pred))

evaluate_model("Logistic Regression", y_test, log_preds)
evaluate_model("Decision Tree", y_test, tree_preds)

# Confusion Matrix Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.heatmap(confusion_matrix(y_test, log_preds), annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title("Logistic Regression Confusion Matrix")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")

sns.heatmap(confusion_matrix(y_test, tree_preds), annot=True, fmt='d', cmap='Greens', ax=axes[1])
axes[1].set_title("Decision Tree Confusion Matrix")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Actual")
plt.tight_layout()
plt.show()

# ========== 🧪 Predict from User Input ==========
def get_user_input():
    print("\n🧾 Enter New Patient Details:")
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

# Predict for new user input
user_input = get_user_input()

# Predict using both models
log_pred = log_model.predict(user_input)[0]
tree_pred = tree_model.predict(user_input)[0]

print("\n🧠 Logistic Regression Prediction:", "Diabetic" if log_pred == 1 else "Not Diabetic")
print("🌳 Decision Tree Prediction:", "Diabetic" if tree_pred == 1 else "Not Diabetic")
