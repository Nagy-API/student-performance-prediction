from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Literal
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
import joblib


# Load trained pipeline
model = joblib.load("../model/student_pass_prediction_pipeline.pkl")


# Create FastAPI app
app = FastAPI(
    title="Student Performance Prediction API",
    description="An API that predicts whether a student is likely to pass or not.",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Input data schema
class StudentData(BaseModel):
    study_hours_per_week: float = Field(
    ...,
    ge=0,
    le=40,
    description="Study hours per week, from 0 to 40"
)
    attendance_rate: float = Field(..., ge=0, le=100, description="Attendance rate from 0 to 100")
    previous_grades: float = Field(..., ge=0, le=100, description="Previous grades from 0 to 100")
    participation_in_extracurricular_activities: Literal["Yes", "No"]
    parent_education_level: Literal[
        "High School",
        "Associate",
        "Bachelor",
        "Master",
        "Doctorate"
    ]


@app.get("/")
def home():
    return {
        "message": "Student Performance Prediction API is running."
    }


@app.post("/predict")
def predict(data: StudentData):
    # Convert input data to DataFrame with the same column names used during training
    input_data = pd.DataFrame([{
        "Study Hours per Week": data.study_hours_per_week,
        "Attendance Rate": data.attendance_rate,
        "Previous Grades": data.previous_grades,
        "Participation in Extracurricular Activities": data.participation_in_extracurricular_activities,
        "Parent Education Level": data.parent_education_level
    }])

    prediction = model.predict(input_data)[0]

    result = "Passed" if prediction == 1 else "Not Passed"

    return {
        "prediction": int(prediction),
        "result": result
    }
