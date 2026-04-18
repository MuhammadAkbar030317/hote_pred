# import pandas as pd
# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from typing import Optional
# import joblib
# from pathlib import Path

# # ===== Model yuklash =====
# BASE_DIR = Path(__file__).resolve().parent.parent
# MODEL_PATH = BASE_DIR / "Models" / "best" / "best_model_pipeline.joblib"

# try:
#     loaded = joblib.load(MODEL_PATH)
#     preprocessor = loaded["preprocessor"]
#     model = loaded["model"]
# except Exception as e:
#     raise RuntimeError(f"Model yuklanmadi: {e}")

# app = FastAPI(title="Hotel Booking Cancellation API", version="1.0")

# # ===== Input schema =====
# class BookingInput(BaseModel):
#     hotel: str
#     lead_time: int
#     arrival_date_year: int
#     arrival_date_month: str
#     arrival_date_week_number: int
#     arrival_date_day_of_month: int
#     stays_in_weekend_nights: int
#     stays_in_week_nights: int
#     adults: int
#     children: Optional[float] = None
#     babies: int
#     meal: str
#     country: Optional[str] = None
#     market_segment: str
#     distribution_channel: str
#     is_repeated_guest: int
#     previous_cancellations: int
#     previous_bookings_not_canceled: int
#     reserved_room_type: str
#     assigned_room_type: str
#     booking_changes: int
#     deposit_type: str
#     agent: Optional[float] = None
#     company: Optional[float] = None
#     days_in_waiting_list: int
#     customer_type: str
#     adr: float
#     required_car_parking_spaces: int
#     total_of_special_requests: int
#     city: Optional[str] = None


# @app.get("/")
# def root():
#     return {"status": "Hotel Cancellation API is running"}


# @app.get("/health")
# def health():
#     return {"status": "ok"}



# @app.post("/predict")
# def predict(data: BookingInput):
#     try:
#         df = pd.DataFrame([data.model_dump()])
#         X = preprocessor.transform(df)
#         prediction = int(model.predict(X)[0])

#         if hasattr(model, "predict_proba"):
#             proba = model.predict_proba(X)[0][1]
#         else:
#             proba = None

#         return {
#             "is_canceled": prediction,
#             "cancellation_probability": round(float(proba), 4) if proba is not None else None,
#             "status": "Canceled" if prediction == 1 else "Not Canceled"
#         }
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))




import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import joblib
import numpy as np
from pathlib import Path

# ===== Model yuklash =====
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "Models" / "best" / "best_model_pipeline.joblib"

try:
    loaded = joblib.load(MODEL_PATH)
    preprocessor = loaded["preprocessor"]
    selected_features_idx = loaded["selected_features_idx"]
    model = loaded["model"]
except Exception as e:
    raise RuntimeError(f"Model yuklanmadi: {e}")

app = FastAPI(title="Hotel Booking Cancellation API", version="1.0")

# ===== Input schema =====
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
    city: Optional[str] = None


@app.get("/")
def root():
    return {"status": "Hotel Cancellation API is running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(data: BookingInput):
    try:
        df = pd.DataFrame([data.model_dump()])

        X = preprocessor.transform(df)
        X_selected = X[:, selected_features_idx]  # ✅ Feature selection qo'llash
        prediction = int(model.predict(X_selected)[0])

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_selected)[0][1]
        else:
            proba = None

        return {
            "is_canceled": prediction,
            "cancellation_probability": round(float(proba), 4) if proba is not None else None,
            "status": "Canceled" if prediction == 1 else "Not Canceled"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))