from pathlib import Path
from typing import Literal
import io
import os

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sklearn.preprocessing import FunctionTransformer


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "churn_model.pkl"
NUMERIC_FEATURES = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

app = FastAPI(title="Customer Churn API", version="1.0.0")
frontend_origins = [
    origin.strip().rstrip("/")
    for origin in os.getenv("FRONTEND_ORIGINS", "").split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", *frontend_origins],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

model = None
model_error = None
try:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH.name}")
    model = joblib.load(MODEL_PATH)
    # In this older pickle, ColumnTransformer saved the numeric passthrough
    # transformer as the literal string "passthrough". Newer sklearn versions
    # expect a transformer object during transform, so restore its identity
    # behavior after loading without changing the trained model parameters.
    preprocessor = model.named_steps.get("preprocessor")
    if preprocessor is not None and hasattr(preprocessor, "transformers_"):
        preprocessor.transformers_ = [
            (name, FunctionTransformer(validate=False) if transformer == "passthrough" else transformer, columns)
            for name, transformer, columns in preprocessor.transformers_
        ]
except Exception as exc:
    model_error = f"{type(exc).__name__}: {exc}"


class Customer(BaseModel):
    SeniorCitizen: Literal[0, 1]
    tenure: int = Field(ge=0, le=72)
    MonthlyCharges: float = Field(ge=18.25, le=118.75)
    TotalCharges: float = Field(ge=0, le=10000)
    gender: Literal["Female", "Male"]
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check", "Mailed check", "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]


def readable_feature(name: str) -> str:
    name = name.removeprefix("num__").removeprefix("cat__")
    replacements = {
        "Contract_Month-to-month": "Month-to-month contract",
        "Contract_One year": "One-year contract",
        "Contract_Two year": "Two-year contract",
        "PaymentMethod_Electronic check": "Electronic check payment",
        "PaymentMethod_Credit card (automatic)": "Automatic credit card payment",
        "PaymentMethod_Bank transfer (automatic)": "Automatic bank transfer",
        "PaymentMethod_Mailed check": "Mailed check payment",
        "InternetService_Fiber optic": "Fiber optic internet",
        "InternetService_DSL": "DSL internet",
        "InternetService_No": "No internet service",
        "OnlineSecurity_Yes": "Online security enabled",
        "OnlineSecurity_No": "No online security",
        "OnlineBackup_Yes": "Online backup enabled",
        "OnlineBackup_No": "No online backup",
        "DeviceProtection_Yes": "Device protection enabled",
        "DeviceProtection_No": "No device protection",
        "TechSupport_Yes": "Technical support enabled",
        "TechSupport_No": "No technical support",
        "PaperlessBilling_Yes": "Paperless billing",
        "PaperlessBilling_No": "No paperless billing",
        "PhoneService_Yes": "Phone service",
        "PhoneService_No": "No phone service",
        "MultipleLines_Yes": "Multiple phone lines",
        "MultipleLines_No": "No multiple lines",
        "MultipleLines_No phone service": "No phone service",
        "Dependents_Yes": "Has dependents",
        "Dependents_No": "No dependents",
        "Partner_Yes": "Has a partner",
        "Partner_No": "No partner",
        "StreamingTV_Yes": "Streaming TV enabled",
        "StreamingTV_No": "No streaming TV",
        "StreamingMovies_Yes": "Streaming movies enabled",
        "StreamingMovies_No": "No streaming movies",
        "gender_Male": "Male",
        "gender_Female": "Female",
        "SeniorCitizen": "Senior citizen",
        "tenure": "Customer tenure",
        "MonthlyCharges": "Monthly charges",
        "TotalCharges": "Total charges",
    }
    return replacements.get(name, name.replace("_", " "))


def explain_prediction(row: pd.DataFrame):
    try:
        preprocessor = model.named_steps["preprocessor"]
        classifier = model.named_steps["classifier"]
        categorical_encoder = next(
            transformer for name, transformer, _ in preprocessor.transformers_ if name == "cat"
        )
        encoded_names = [
            f"{feature}_{category}"
            for feature, categories in zip(CATEGORICAL_FEATURES, categorical_encoder.categories_)
            for category in categories
        ]
        feature_names = [f"num__{name}" for name in NUMERIC_FEATURES] + [
            f"cat__{name}" for name in encoded_names
        ]
        transformed = preprocessor.transform(row)
        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()
        contributions = np.asarray(transformed)[0] * np.asarray(classifier.coef_)[0]
        ranked = sorted(zip(feature_names, contributions), key=lambda item: item[1], reverse=True)
        toward = [{"name": readable_feature(str(name)), "impact": round(float(value), 3)} for name, value in ranked if value > 0][:5]
        away = [{"name": readable_feature(str(name)), "impact": round(float(value), 3)} for name, value in reversed(ranked) if value < 0][:5]
        return toward, away
    except (AttributeError, KeyError, TypeError, ValueError):
        return [], []


def get_recommendations(row: pd.DataFrame, risk: str, risk_factors: list[dict]):
    if risk == "Low":
        return ["Maintain regular engagement and continue monitoring customer satisfaction."]
    factors = {item["name"] for item in risk_factors}
    recommendations = []
    if "Month-to-month contract" in factors:
        recommendations.append("Consider a personalized incentive to move the customer to a longer-term contract.")
    if "Electronic check payment" in factors:
        recommendations.append("Encourage an automatic payment method if it suits the customer.")
    if "No online security" in factors:
        recommendations.append("Explain available online security services and their benefits.")
    if "No technical support" in factors:
        recommendations.append("Offer technical support or a plan that includes support.")
    if "Fiber optic internet" in factors:
        recommendations.append("Review the internet plan and offer suitable service or pricing options.")
    if "Monthly charges" in factors:
        recommendations.append("Review the current plan for a suitable pricing or plan adjustment.")
    if "Paperless billing" in factors:
        recommendations.append("Review billing preferences and explain available billing options.")
    if row.iloc[0]["tenure"] < 12:
        recommendations.append("Provide additional onboarding and early-stage customer support.")
    return recommendations or ["Offer personalized engagement and monitor future churn risk."]


@app.get("/health")
def health():
    return {"status": "ok" if model is not None else "model_unavailable", "model_loaded": model is not None, "detail": model_error}


def positive_class_index():
    classes = list(model.classes_)
    return next((i for i, value in enumerate(classes) if str(value).strip().lower() in {"1", "yes", "true", "churn"}), len(classes) - 1)


def risk_level(probability: float) -> str:
    return "Low" if probability < 0.30 else "Medium" if probability < 0.60 else "High"


@app.post("/predict")
def predict(customer: Customer):
    if model is None:
        raise HTTPException(status_code=503, detail=f"Model could not be loaded. Check backend setup: {model_error}")
    row = pd.DataFrame([customer.model_dump()], columns=FEATURES)
    try:
        predicted = model.predict(row)[0]
        probabilities = model.predict_proba(row)[0]
        churn_index = positive_class_index()
        probability = float(probabilities[churn_index])
        churn = str(predicted).strip().lower() in {"1", "yes", "true", "churn"}
        risk = risk_level(probability)
        risk_factors, protective_factors = explain_prediction(row)
        return {
            "churn": churn,
            "label": "Likely to churn" if churn else "Likely to stay",
            "prediction": "Churn" if churn else "No Churn",
            "churn_probability": round(probability * 100, 2),
            "risk_level": risk,
            "risk_factors": risk_factors,
            "protective_factors": protective_factors,
            "recommendations": get_recommendations(row, risk, risk_factors),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {type(exc).__name__}: {exc}") from exc


@app.post("/predict-csv")
async def predict_csv(request: Request):
    if model is None:
        raise HTTPException(status_code=503, detail=f"Model could not be loaded. Check backend setup: {model_error}")
    filename = request.headers.get("x-filename", "")
    if not filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Upload a .csv file.")
    try:
        frame = pd.read_csv(io.BytesIO(await request.body()))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read CSV: {exc}") from exc

    frame.columns = [str(column).strip() for column in frame.columns]
    missing = [column for column in FEATURES if column not in frame.columns]
    if missing:
        raise HTTPException(status_code=422, detail=f"CSV is missing required model columns: {', '.join(missing)}")
    if frame.empty:
        raise HTTPException(status_code=422, detail="The CSV has no customer rows.")
    if len(frame) > 10000:
        raise HTTPException(status_code=413, detail="Upload up to 10,000 customer rows per file.")

    rows = frame[FEATURES].copy()
    raw_total = rows["TotalCharges"]
    total_text = raw_total.astype("string").str.strip()
    total_numeric = pd.to_numeric(raw_total, errors="coerce")
    invalid_total = total_numeric.isna() & raw_total.notna() & total_text.ne("")
    if invalid_total.any():
        first_row = int(np.flatnonzero(invalid_total.to_numpy())[0]) + 2
        raise HTTPException(status_code=422, detail=f"TotalCharges must be numeric (invalid value near CSV row {first_row}).")
    rows["TotalCharges"] = total_numeric.fillna(0)

    for column in ["SeniorCitizen", "tenure", "MonthlyCharges"]:
        converted = pd.to_numeric(rows[column], errors="coerce")
        if converted.isna().any():
            first_row = int(np.flatnonzero(converted.isna().to_numpy())[0]) + 2
            raise HTTPException(status_code=422, detail=f"{column} must contain a number (invalid value near CSV row {first_row}).")
        rows[column] = converted
    if not rows["SeniorCitizen"].isin([0, 1]).all():
        raise HTTPException(status_code=422, detail="SeniorCitizen values must be 0 or 1.")
    if not rows["tenure"].between(0, 72).all():
        raise HTTPException(status_code=422, detail="Tenure values must be from 0 to 72 months.")
    if not rows["MonthlyCharges"].between(18.25, 118.75).all():
        raise HTTPException(status_code=422, detail="MonthlyCharges values must be between 18.25 and 118.75.")

    encoder = next(transformer for name, transformer, _ in model.named_steps["preprocessor"].transformers_ if name == "cat")
    for column, valid_values in zip(CATEGORICAL_FEATURES, encoder.categories_):
        values = rows[column].astype("string").str.strip()
        if values.isna().any() or values.eq("").any():
            first_row = int(np.flatnonzero((values.isna() | values.eq("")).to_numpy())[0]) + 2
            raise HTTPException(status_code=422, detail=f"{column} is blank (invalid value near CSV row {first_row}).")
        unknown = ~values.isin(valid_values)
        if unknown.any():
            first_row = int(np.flatnonzero(unknown.to_numpy())[0]) + 2
            raise HTTPException(status_code=422, detail=f"Unexpected {column} value near CSV row {first_row}. Use the categories from the analysis dataset.")
        rows[column] = values

    try:
        predictions = model.predict(rows)
        probabilities = model.predict_proba(rows)[:, positive_class_index()]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"CSV prediction failed: {type(exc).__name__}: {exc}") from exc

    if "Name" in frame.columns:
        customer_names = frame["Name"].astype("string").fillna("").str.strip()
    elif "name" in frame.columns:
        customer_names = frame["name"].astype("string").fillna("").str.strip()
    elif "customerID" in frame.columns:
        customer_names = frame["customerID"].astype("string").fillna("").str.strip()
    else:
        customer_names = pd.Series([""] * len(frame))
    results = []
    for index, (predicted, probability) in enumerate(zip(predictions, probabilities)):
        churn = str(predicted).strip().lower() in {"1", "yes", "true", "churn"}
        customer = rows.iloc[index]
        row_frame = rows.iloc[[index]]
        probability = float(probability)
        risk = risk_level(probability)
        risk_factors, protective_factors = explain_prediction(row_frame)
        results.append({
            "name": customer_names.iloc[index] or f"Customer {index + 1}",
            "churn": churn,
            "prediction": "Churn" if churn else "No Churn",
            "churn_probability": round(probability * 100, 2),
            "risk_level": risk,
            "risk_factors": risk_factors,
            "protective_factors": protective_factors,
            "recommendations": get_recommendations(row_frame, risk, risk_factors),
        })
    return {"count": len(results), "results": results}
