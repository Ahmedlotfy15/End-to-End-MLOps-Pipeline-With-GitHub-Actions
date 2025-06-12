import numpy as np
import pandas as pd
import joblib
import onnxruntime


session = onnxruntime.InferenceSession('models/purchase_nn.onnx')

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

scaler = joblib.load('preprocessors/scaler.pkl')
encoder = joblib.load('preprocessors/encoder.pkl')

def predict_purchase(days_on_platform, minutes_watched, courses_started, practice_exams_passed, minutes_spent_on_exams, student_country):

    input_df = pd.DataFrame({
        'days_on_platform': [days_on_platform],
        'minutes_watched': [minutes_watched],
        'courses_started': [courses_started],
        'practice_exams_passed': [practice_exams_passed],
        'minutes_spent_on_exams': [minutes_spent_on_exams],
        'student_country': [student_country]
    })

    input_df['student_country_enc'] = encoder.transform(input_df[['student_country']])

    input_df = input_df.drop(columns=['student_country'])

    input_array = input_df.to_numpy()

    input_scaled = scaler.transform(input_array).astype(np.float32)  # Ensure float32 for ONNX

    # Run inference using ONNX session
    ort_inputs = {input_name: input_scaled}
    ort_outputs = session.run([output_name], ort_inputs)
    
    # Get predictions
    output = ort_outputs[0]
    proba = 1 / (1 + np.exp(-output)) 

    prediction = (proba >= 0.7).astype(int)  

    return prediction[0][0], proba[0][0]  
result, prob = predict_purchase(
    days_on_platform=10,         
    minutes_watched=500,         
    courses_started=10,          
    practice_exams_passed=10,    
    minutes_spent_on_exams=10,   
    student_country='egypt'       
)

