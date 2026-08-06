import joblib
import pandas as pd
from pathlib import Path
from fastapi import APIRouter
from db.schema import EmployeeSchema
from db.columns import columns

BASE_DIR = Path(__file__).parent.parent

model  = joblib.load(BASE_DIR / "model_nei_HR_Employee.pkl")
scaler = joblib.load(BASE_DIR / "scaler_HR_Employee.pkl")

predict_rf_router = APIRouter(prefix="/predict_rf", tags=["Predict_RF"])


def build_features(data: EmployeeSchema) -> pd.DataFrame:
    d = data.model_dump()

    # Числовые / бинарные поля — берём напрямую
    row: dict = {
        "Age":                      d["Age"],
        "DailyRate":                d.get("DailyRate", 0),
        "DistanceFromHome":         d["DistanceFromHome"],
        "Education":                d.get("Education", 0),
        "EnvironmentSatisfaction":  d["EnvironmentSatisfaction"],
        "HourlyRate":               d.get("HourlyRate", 0),
        "JobInvolvement":           d.get("JobInvolvement", 0),
        "JobLevel":                 d.get("JobLevel", 0),
        "JobSatisfaction":          d["JobSatisfaction"],
        "MonthlyIncome":            d["MonthlyIncome"],
        "MonthlyRate":              d.get("MonthlyRate", 0),
        "NumCompaniesWorked":       d.get("NumCompaniesWorked", 0),
        "PercentSalaryHike":        d.get("PercentSalaryHike", 0),
        "PerformanceRating":        d.get("PerformanceRating", 0),
        "RelationshipSatisfaction": d.get("RelationshipSatisfaction", 0),
        "StockOptionLevel":         d.get("StockOptionLevel", 0),
        "TotalWorkingYears":        d.get("TotalWorkingYears", 0),
        "TrainingTimesLastYear":    d.get("TrainingTimesLastYear", 0),
        "WorkLifeBalance":          d["WorkLifeBalance"],
        "YearsAtCompany":           d["YearsAtCompany"],
        "YearsInCurrentRole":       d.get("YearsInCurrentRole", 0),
        "YearsSinceLastPromotion":  d.get("YearsSinceLastPromotion", 0),
        "YearsWithCurrManager":     d.get("YearsWithCurrManager", 0),
        "Attrition_num":            0,

        # OverTime → бинарный
        "OverTime": 1 if d["OverTime"] == "Yes" else 0,

        # Gender (drop_first → Female удалён, остался Male)
        "Gender": 1 if d.get("Gender") == "Male" else 0,

        # BusinessTravel (drop_first → Non-Travel удалён)
        "BusinessTravel": (
            1 if d.get("BusinessTravel") == "Travel_Frequently"
            else (2 if d.get("BusinessTravel") == "Travel_Rarely" else 0)
        ),
    }

    # One-Hot: JobRole (drop_first → Healthcare Representative удалён)
    for role in [
        "Human Resources", "Laboratory Technician", "Manager",
        "Manufacturing Director", "Research Director", "Research Scientist",
        "Sales Executive", "Sales Representative",
    ]:
        row[f"JobRole_{role}"] = 1 if d["JobRole"] == role else 0

    # One-Hot: Department (drop_first → Human Resources удалён)
    row["Department_Research & Development"] = 1 if d.get("Department") == "Research & Development" else 0
    row["Department_Sales"]                  = 1 if d.get("Department") == "Sales" else 0

    # One-Hot: MaritalStatus (drop_first → Divorced удалён)
    row["MaritalStatus_Married"] = 1 if d.get("MaritalStatus") == "Married" else 0
    row["MaritalStatus_Single"]  = 1 if d.get("MaritalStatus") == "Single"  else 0

    # One-Hot: EducationField (drop_first → Human Resources удалён)
    for field in ["Life Sciences", "Marketing", "Medical", "Other", "Technical Degree"]:
        row[f"EducationField_{field}"] = 1 if d.get("EducationField") == field else 0

    return pd.DataFrame([{col: row.get(col, 0) for col in columns}], columns=columns)


@predict_rf_router.post("/")
async def check_rf(data: EmployeeSchema):
    df_input   = build_features(data)
    scaled     = scaler.transform(df_input)
    prediction = model.predict(scaled)[0]
    label      = "Сотрудник, скорее всего, уволится." if prediction == 1 else "Сотрудник, скорее всего, останется в компании."
    return {"Attrition": label}