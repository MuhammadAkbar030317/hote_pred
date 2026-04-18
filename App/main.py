import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "Models" / "hotel_cancellation_pipeline.joblib"

try:
    loaded = joblib.load(MODEL_PATH)
    pipeline = loaded["pipeline"]
except Exception as e:
    raise RuntimeError(f"Model yuklanmadi: {e}")

app = FastAPI(title="Hotel Booking Cancellation API", version="1.0")

class BookingInput(BaseModel):
    hotel: str
    lead_time: int
    arrival_date_year: int
    arrival_date_month: str
    arrival_date_week_number: int
    arrival_date_day_of_month: int
    stays_in_weekend_nights: int
    stays_in_week_nights: int
    adults: int
    children: Optional[float] = None
    babies: int
    meal: str
    country: Optional[str] = None
    market_segment: str
    distribution_channel: str
    is_repeated_guest: int
    previous_cancellations: int
    previous_bookings_not_canceled: int
    reserved_room_type: str
    assigned_room_type: str
    booking_changes: int
    deposit_type: str
    agent: Optional[float] = None
    company: Optional[float] = None
    days_in_waiting_list: int
    customer_type: str
    adr: float
    required_car_parking_spaces: int
    total_of_special_requests: int

@app.get("/")
def root():
    return {"status": "Hotel Cancellation API is running"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(data: BookingInput):
    df = pd.DataFrame([data.model_dump()])
    prediction = int(pipeline.predict(df)[0])

    if hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba(df)[0][1]
    else:
        proba = None

    return {
        "is_canceled": prediction,
        "cancellation_probability": round(float(proba), 4) if proba is not None else None,
        "status": "Canceled" if prediction == 1 else "Not Canceled"
    }