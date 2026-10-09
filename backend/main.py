from pathlib import Path
from typing import Literal
import io
import json
import os
import urllib.error
import urllib.request

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


def generate_ai_analysis(customer: Customer, prediction: dict) -> dict:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise HTTPException(status_code=503, detail="AI explanations are not configured. Add GEMINI_API_KEY to the backend environment.")

    # Do not send demographic fields to the external AI provider. They remain
    # available to the saved ML model, but are not used to explain a result.
    customer_features = customer.model_dump(exclude={"gender", "SeniorCitizen"})
    model_name = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite").strip()
    request_body = {
        "systemInstruction": {
            "parts": [{"text": (
                "You explain a customer churn model result in plain, respectful language. "
                "Use only the provided non-demographic account/service features and the model output. "
                "Do not claim any feature caused churn or that the prediction is certain. "
                "Do not invent information. Give a concise explanation and 2 or 3 practical, "
                "non-coercive follow-up suggestions. Return JSON with string key 'summary' and "
                "array-of-strings key 'suggested_steps'."
            )}],
        },
        "contents": [{"parts": [{"text": json.dumps({
            "customer_features": customer_features,
            "model_output": prediction,
            "task": "Explain this estimate and suggest suitable retention follow-up.",
        })}]}],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 300,
            "responseMimeType": "application/json",
        },
    }
    request = urllib.request.Request(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent",
        data=json.dumps(request_body).encode("utf-8"),
        headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=35) as response:
            payload = json.loads(response.read().decode("utf-8"))
        content = payload["candidates"][0]["content"]["parts"][0]["text"]
        generated = json.loads(content)
        summary = generated.get("summary")
        steps = generated.get("suggested_steps")
        if not isinstance(summary, str) or not summary.strip() or not isinstance(steps, list):
            raise ValueError("The AI response did not contain the expected fields.")
        steps = [step.strip() for step in steps if isinstance(step, str) and step.strip()][:3]
        return {"summary": summary.strip(), "suggested_steps": steps}
    except HTTPException:
        raise
    except urllib.error.HTTPError as exc:
        if exc.code == 429:
            raise HTTPException(status_code=429, detail="Gemini's free request limit was reached. Wait and try again later.") from exc
        raise HTTPException(status_code=502, detail="Gemini rejected the request. Check the backend AI configuration.") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise HTTPException(status_code=502, detail="Gemini could not be reached. Try again shortly.") from exc
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=502, detail="Gemini returned an invalid explanation. Try again.") from exc


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
        return {
            "churn": churn,
            "label": "Likely to churn" if churn else "Likely to stay",
            "prediction": "Churn" if churn else "No Churn",
            "churn_probability": round(probability * 100, 2),
            "risk_level": risk,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {type(exc).__name__}: {exc}") from exc


@app.post("/explain")
def explain_customer(customer: Customer):
    if model is None:
        raise HTTPException(status_code=503, detail=f"Model could not be loaded. Check backend setup: {model_error}")
    row = pd.DataFrame([customer.model_dump()], columns=FEATURES)
    try:
        predicted = model.predict(row)[0]
        probability = float(model.predict_proba(row)[0][positive_class_index()])
        churn = str(predicted).strip().lower() in {"1", "yes", "true", "churn"}
        prediction = {
            "prediction": "Churn" if churn else "No Churn",
            "risk_level": risk_level(probability),
            "churn_probability": round(probability * 100, 2),
        }
        return generate_ai_analysis(customer, prediction)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"AI explanation failed: {type(exc).__name__}: {exc}") from exc


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
        row_frame = rows.iloc[[index]]
        probability = float(probability)
        risk = risk_level(probability)
        results.append({
            "name": customer_names.iloc[index] or f"Customer {index + 1}",
            "churn": churn,
            "prediction": "Churn" if churn else "No Churn",
            "churn_probability": round(probability * 100, 2),
            "risk_level": risk,
            "customer_features": json.loads(row_frame.iloc[0].to_json()),
        })
    return {"count": len(results), "results": results}
