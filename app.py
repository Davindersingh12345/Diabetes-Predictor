# ========== 📦 Import libraries ==========
from flask import Flask, request, render_template, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ========== ⚙️ Flask App Setup ==========
app = Flask(__name__)

# ========== 📥 Load dataset ==========
df = pd.read_csv(r"C:\Users\ASUS\Downloads\diabetes_prediction_ - diabetes_prediction_impure.csv.csv")

# Data Cleaning
df_clean = df.dropna(subset=['Diabetic'])
for col in ['Age', 'BMI', 'Glucose', 'BloodPressure']:
    df_clean[col].fillna(df_clean[col].median(), inplace=True)
df_clean['FamilyHistory'] = df_clean['FamilyHistory'].map({'Yes': 1, 'No': 0})
df_clean['Diabetic'] = (df_clean['Diabetic'] >= 1).astype(int)

# ========== 📊 EDA (Optional, run once) ==========
if not os.path.exists("static/eda"):
    os.makedirs("static/eda")
    # Correlation Heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(df_clean.corr(), annot=True, cmap='coolwarm')
    plt.title("Correlation Heatmap")
    plt.savefig("static/eda/corr_heatmap.png")
    plt.close()

    # Countplot
    sns.countplot(x='Diabetic', data=df_clean, palette='Set2')
    plt.title("Count of Diabetic vs Non-Diabetic")
    plt.savefig("static/eda/diabetic_count.png")
    plt.close()

# ========== 🤖 Model Training ==========
X = df_clean.drop('Diabetic', axis=1)
y = df_clean['Diabetic']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

log_model = LogisticRegression()
log_model.fit(X_train, y_train)

tree_model = DecisionTreeClassifier(random_state=42)
tree_model.fit(X_train, y_train)

# Store column order for prediction
feature_columns = list(X.columns)

# ========== 🌐 Web Routes ==========

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Preprocess input in same order and type as training
        input_data = {
            'Age': float(data['Age']),
            'BMI': float(data['BMI']),
            'Glucose': float(data['Glucose']),
            'BloodPressure': float(data['BloodPressure']),
            'FamilyHistory': int(data['FamilyHistory'])
        }

        input_df = pd.DataFrame([input_data])[feature_columns]
        print(input_df)
        # Predict with both models
        log_pred = log_model.predict(input_df)[0]
        tree_pred = tree_model.predict(input_df)[0]

        return jsonify({
            "logistic_prediction": int(log_pred),
            "tree_prediction": int(tree_pred)
        })

    except Exception as e:
        print("Prediction error:", e)
        return jsonify({"error": "Prediction failed", "details": str(e)})

# ========== 🚀 Run Server ==========
if __name__ == "__main__":
    app.run(debug=True)
