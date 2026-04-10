import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib
import json
import os

def main():
    print("Loading data...")
    columns = [
        "age", "sex", "chest_pain", "resting_bp", "cholesterol",
        "fasting_bs", "rest_ecg", "max_hr", "exercise_angina",
        "oldpeak", "slope", "num_vessels", "thal", "target"
    ]
    
    # Read the data
    df = pd.read_csv('statlog+heart/heart.dat', sep=' ', names=columns)
    
    # Map target variable
    # 1 = No disease -> 0
    # 2 = Disease -> 1
    df['target'] = df['target'].map({1: 0, 2: 1})
    
    # Apply One-Hot Encoding
    df = pd.get_dummies(df, columns=[
        'chest_pain', 'rest_ecg', 'slope', 'thal'
    ], drop_first=True)
    
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Save the expected columns after get_dummies to ensure matching input locally
    expected_columns = list(X.columns)
    with open("expected_columns.json", "w") as f:
        json.dump(expected_columns, f)
    print("Saved expected_columns.json")
    
    # Scale Data
    print("Scaling and splitting data...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    # Train model
    print("Training Logistic Regression...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    accuracy = model.score(X_test, y_test)
    print(f"Model test accuracy: {accuracy:.4f}")
    
    # Export artifacts
    joblib.dump(model, 'logistic_model.joblib')
    joblib.dump(scaler, 'scaler.joblib')
    print("Saved logistic_model.joblib and scaler.joblib successfully.")

if __name__ == "__main__":
    main()
