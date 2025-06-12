from fastapi import FastAPI
from pydantic import BaseModel
from infer import predict_purchase

app = FastAPI(
    title="Student Purchase Prediction API",
    description="API for predicting student purchase probability",
)

class StudentData(BaseModel):
    days_on_platform: int
    minutes_watched: float
    courses_started: int
    practice_exams_passed: int
    minutes_spent_on_exams: float
    student_country: str

@app.post("/predict")
async def predict(data: StudentData):
    prediction, probability = predict_purchase(
        days_on_platform=data.days_on_platform,
        minutes_watched=data.minutes_watched,
        courses_started=data.courses_started,
        practice_exams_passed=data.practice_exams_passed,
        minutes_spent_on_exams=data.minutes_spent_on_exams,
        student_country=data.student_country
    )
    
    return {
        "prediction": int(prediction),
        "probability": float(probability)
    }

