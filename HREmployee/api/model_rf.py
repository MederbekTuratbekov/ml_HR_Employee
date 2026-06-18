import joblib
import pandas as pd
from pathlib import Path
from fastapi import APIRouter
from db.schema import EmployeeSchema
from db.columns import columns

BASE_DIR = Path(__file__).parent.parent

model = joblib.load(BASE_DIR / 'model_rf_HR_Employee.pkl')
scaler = joblib.load(BASE_DIR / 'scaler_HR_Employee.pkl')

predict_rf_router = APIRouter(prefix='/predict_rf', tags=['Predict_RF'])

@predict_rf_router.post('/')
async def check_rf(data: EmployeeSchema):
    data_dict = dict(data)

    # 1. Бинаризация OverTime
    new_over_time = data_dict.pop('OverTime')
    data_dict['OverTime'] = 1 if new_over_time == 'Yes' else 0

    # 2. One-Hot Encoding для JobRole
    new_job_role = data_dict.pop('JobRole')
    roles = [
        'Human Resources', 'Laboratory Technician', 'Manager',
        'Manufacturing Director', 'Research Director', 'Research Scientist',
        'Sales Executive', 'Sales Representative'
    ]
    for role in roles:
        data_dict[f'JobRole_{role}'] = 1 if new_job_role == role else 0

    data_dict['Attrition_num'] = 0

    full_features_dict = {col: data_dict.get(col, 0) for col in columns}

    df_input = pd.DataFrame([full_features_dict], columns=columns)

    scaled = scaler.transform(df_input)
    prediction = model.predict(scaled)

    return {"Attrition": "Сотрудник, скорее всего, уволится." if prediction[0] == 1 else "Сотрудник, скорее всего, останется в компании."}
