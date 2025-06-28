# 🎓 Student Performance Predictor – EduQuest Coaching

This project is built to help **EduQuest Coaching** predict the final exam scores of students using machine learning. The model takes into account various factors like academic history, engagement levels, attendance rate, and demographic details to provide a personalized performance forecast.

---

## 📁 Dataset Used

**File:** `Student Performance Predictor for EduQuest Coaching.csv`  
**Rows:** 1000  
**Target Variable:** `final_exam_score`

### 📊 Key Features
- `gender`
- `age`
- `parental_education`
- `family_income`
- `internet_access`
- `previous_exam_score`
- `attendance_rate`
- `homework_completion_rate`
- `class_participation_score`
- `number_of_absences`
- `extra_curricular_involvement`
- `learning_hours_per_week`
- `tutor_support`

---

## 🤖 Machine Learning Approach

We used a **Random Forest Regressor** to predict the `final_exam_score`. The pipeline includes:

- Label encoding for categorical features
- Feature scaling using StandardScaler
- Model training and evaluation
- Prediction for new student input

---

## 📈 Model Evaluation Metrics

- **MAE (Mean Absolute Error)**: ~ Low (Good)
- **R² Score**: ~ High (Accurate)

*(Exact values will be printed after training in Colab)*

---

## 🛠 How to Run This Project in Google Colab

1. Clone or download this repository.
2. Upload the notebook and CSV to [Google Colab](https://colab.research.google.com/).
3. Run the notebook cells step-by-step.
4. Upload your dataset when prompted.
5. Model will be trained, evaluated, and ready for predictions.

---

## 🧪 Sample Prediction

You can edit the sample `new_student` dictionary in the notebook to test different student profiles. The model will return a predicted **final exam score**.

```python
'gender': 'Male',
'age': 16,
'parental_education': 'Graduate',
'family_income': 35000,
...
