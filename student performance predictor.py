import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
from google.colab import files

# 📁 STEP 2: Upload Single CSV File
print("Upload your dataset (Student Performance Predictor for EduQuest Coaching.csv)...")
uploaded = files.upload()

# 📌 STEP 3: Load the Dataset
df = pd.read_csv("Student Performance Predictor for EduQuest Coaching.csv")

# 📊 STEP 4: Preprocessing
# Label encode categorical columns
label_encoders = {}
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Separate features and target
X = df.drop("final_exam_score", axis=1)
y = df["final_exam_score"]

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 🧠 STEP 5: Train the Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model and scaler
joblib.dump(model, "student_performance_model.pkl")
joblib.dump(scaler, "scaler.pkl")

# 📈 STEP 6: Evaluate Model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Model Evaluation:")
print(f"MAE: {mae:.2f}")
print(f"R² Score: {r2:.2f}")

# 🔮 STEP 7: Predict New Student's Performance
# Example new input (make sure it includes all columns except the target)
new_student = pd.DataFrame([{
    'gender': label_encoders['gender'].transform(['Male'])[0],
    'age': 16,
    'parental_education': label_encoders['parental_education'].transform(['Graduate'])[0],
    'family_income': 35000,
    'internet_access': label_encoders['internet_access'].transform(['Yes'])[0],
    'previous_exam_score': 78.0,
    'attendance_rate': 90.0,
    'homework_completion_rate': 80.0,
    'class_participation_score': 7.5,
    'number_of_absences': 2,
    'extra_curricular_involvement': label_encoders['extra_curricular_involvement'].transform(['Moderate'])[0],
    'learning_hours_per_week': 10.0,
    'tutor_support': label_encoders['tutor_support'].transform(['Yes'])[0]
}])

# Scale the input
new_student_scaled = scaler.transform(new_student)
predicted_score = model.predict(new_student_scaled)
print(f"Predicted Final Exam Score: {predicted_score[0]:.2f}")
