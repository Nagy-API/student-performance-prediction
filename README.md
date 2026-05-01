# Student Performance Prediction

This project uses machine learning to predict whether a student is likely to pass or not based on academic and student-related features.

The project includes data exploration, preprocessing, feature engineering experiments, model training, model evaluation, a saved machine learning pipeline, a FastAPI backend, and a simple web interface.

## Project Overview

The goal of this project is to build a machine learning model that predicts student performance using the following features:

- Study Hours per Week
- Attendance Rate
- Previous Grades
- Participation in Extracurricular Activities
- Parent Education Level

The target variable is:

- Passed

This is a binary classification problem where the model predicts whether a student passed or did not pass.

## Project Workflow

The notebook follows these steps:

1. Import required libraries
2. Load and inspect the dataset
3. Explore missing values and duplicated records
4. Clean invalid numerical values
5. Analyze numerical and categorical features
6. Encode categorical variables
7. Create and evaluate feature engineering ideas
8. Train multiple machine learning models
9. Compare model performance
10. Build a final machine learning pipeline
11. Save the trained pipeline using Joblib
12. Use the saved model in a FastAPI application
13. Create a simple web interface to interact with the API

## Dataset

The dataset contains student-related information such as study hours, attendance rate, previous grades, extracurricular activity participation, parent education level, and pass status.

Some data quality issues were found during exploration, including:

- Missing values
- Negative study hours
- Invalid attendance rates
- Invalid previous grades

These issues were handled before model training.

## Feature Engineering

Several engineered features were tested, including interaction features and academic score-based features.

However, the engineered features did not improve the model performance in a meaningful way. The ROC AUC score stayed close to 0.50, which means the model was performing almost like random guessing.

Because of that, the final model was trained using only the original features.

## Models Tested

The following models were tested:

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost

All models achieved similar performance, with ROC AUC scores close to 0.50. This suggests that the available features do not contain strong predictive signals for accurately separating passed and not-passed students.

## Final Model

The final model is a Logistic Regression model inside a Scikit-learn pipeline.

The pipeline includes:

- Median imputation for numerical features
- Standard scaling for numerical features
- Most frequent imputation for categorical features
- One-hot encoding for categorical features
- Logistic Regression classifier

The final pipeline was saved as:

```text
student_pass_prediction_pipeline.pkl
```

## Project Structure

```text
student-performance-prediction/
│
├── app/
│   ├── main.py
│   └── index.html
│
├── data/
│   └── student_performance_prediction.csv
│
├── model/
│   └── student_pass_prediction_pipeline.pkl
│
├── notebook/
│   └── student-performance-prediction.ipynb
│
├── requirements.txt
├── README.md
└── .gitignore
```

## How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/student-performance-prediction.git
cd student-performance-prediction
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the FastAPI server

From the main project folder, run:

```bash
uvicorn app.main:app --reload
```

The API will run at:

```text
http://127.0.0.1:8000
```

You can open the API documentation here:

```text
http://127.0.0.1:8000/docs
```

### 4. Open the web interface

Open this file in the browser:

```text
app/index.html
```

Then enter the student information and click the prediction button.

## API Example

Example input:

```json
{
  "study_hours_per_week": 10,
  "attendance_rate": 80,
  "previous_grades": 75,
  "participation_in_extracurricular_activities": "Yes",
  "parent_education_level": "Bachelor"
}
```

Example output:

```json
{
  "prediction": 1,
  "result": "Passed"
}
```

## Important Note

The model performance is limited because the available dataset does not show strong predictive relationships between the input features and the target variable.

This project is mainly useful for demonstrating the full machine learning workflow, including data analysis, preprocessing, model training, pipeline creation, API deployment, and frontend integration.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib
- Seaborn
- FastAPI
- HTML
- CSS
- JavaScript

## Kaggle Notebook

The notebook version of this project is also available on Kaggle:

https://www.kaggle.com/code/mlnagy/student-performance-prediction/edit
