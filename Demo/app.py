import joblib
import pandas as pd
import gradio as gr
from pathlib import Path

BASE_DIR   = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "Models" / "best" / "best_model_pipeline.joblib"

# ✅ To'g'ri yuklanish
loaded                = joblib.load(MODEL_PATH)
preprocessor          = loaded["preprocessor"]
selected_features_idx = loaded["selected_features_idx"]
model                 = loaded["model"]


def predict(
    hotel, lead_time, arrival_date_year, arrival_date_month,
    arrival_date_week_number, arrival_date_day_of_month,
    stays_in_weekend_nights, stays_in_week_nights,
    adults, children, babies, meal, country, market_segment,
    distribution_channel, is_repeated_guest, previous_cancellations,
    previous_bookings_not_canceled, reserved_room_type, assigned_room_type,
    booking_changes, deposit_type, agent, company, days_in_waiting_list,
    customer_type, adr, required_car_parking_spaces,
    total_of_special_requests, city
):
    df = pd.DataFrame([{
        "hotel": hotel,
        "lead_time": lead_time,
        "arrival_date_year": arrival_date_year,
        "arrival_date_month": arrival_date_month,
        "arrival_date_week_number": arrival_date_week_number,
        "arrival_date_day_of_month": arrival_date_day_of_month,
        "stays_in_weekend_nights": stays_in_weekend_nights,
        "stays_in_week_nights": stays_in_week_nights,
        "adults": adults,
        "children": children,
        "babies": babies,
        "meal": meal,
        "country": country,
        "market_segment": market_segment,
        "distribution_channel": distribution_channel,
        "is_repeated_guest": is_repeated_guest,
        "previous_cancellations": previous_cancellations,
        "previous_bookings_not_canceled": previous_bookings_not_canceled,
        "reserved_room_type": reserved_room_type,
        "assigned_room_type": assigned_room_type,
        "booking_changes": booking_changes,
        "deposit_type": deposit_type,
        "agent": agent,
        "company": company,
        "days_in_waiting_list": days_in_waiting_list,
        "customer_type": customer_type,
        "adr": adr,
        "required_car_parking_spaces": required_car_parking_spaces,
        "total_of_special_requests": total_of_special_requests,
        "city": city
    }])

    # ✅ To'g'ri predict — uch qadam
    X_processed = preprocessor.transform(df)
    X_selected  = X_processed[:, selected_features_idx]
    pred        = int(model.predict(X_selected)[0])
    proba       = model.predict_proba(X_selected)[0][1]

    label = "🔴 Bekor qiladi" if pred == 1 else "🟢 Bekor qilmaydi"
    return f"{label} | Ehtimollik: {proba:.4f}"


# ===== Gradio UI =====
demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Textbox(label="Hotel"),
        gr.Number(label="Lead Time"),
        gr.Number(label="Arrival Year"),
        gr.Textbox(label="Arrival Month"),
        gr.Number(label="Arrival Week Number"),
        gr.Number(label="Arrival Day"),
        gr.Number(label="Weekend Nights"),
        gr.Number(label="Week Nights"),
        gr.Number(label="Adults"),
        gr.Number(label="Children"),
        gr.Number(label="Babies"),
        gr.Textbox(label="Meal"),
        gr.Textbox(label="Country"),
        gr.Textbox(label="Market Segment"),
        gr.Textbox(label="Distribution Channel"),
        gr.Number(label="Repeated Guest (0/1)"),
        gr.Number(label="Previous Cancellations"),
        gr.Number(label="Previous Bookings Not Canceled"),
        gr.Textbox(label="Reserved Room Type"),
        gr.Textbox(label="Assigned Room Type"),
        gr.Number(label="Booking Changes"),
        gr.Textbox(label="Deposit Type"),
        gr.Number(label="Agent"),
        gr.Number(label="Company"),
        gr.Number(label="Days in Waiting List"),
        gr.Textbox(label="Customer Type"),
        gr.Number(label="ADR"),
        gr.Number(label="Car Parking Spaces"),
        gr.Number(label="Special Requests"),
        gr.Textbox(label="City"),
    ],
    outputs="text",
    title="🏨 Hotel Booking Cancellation Prediction"
)

demo.launch()