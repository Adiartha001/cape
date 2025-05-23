#!/usr/bin/env python
# coding: utf-8

# In[7]:


import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load model dan encoder
model = joblib.load('XG_booking_status.pkl')
booking_status_encode = joblib.load('booking_status_encode.pkl')  # DICT, bukan LabelEncoder
oneHot_encode_room = joblib.load('oneHot_encode_room.pkl')
oneHot_encode_meal = joblib.load('oneHot_encode_meal.pkl')
oneHot_encode_mark = joblib.load('oneHot_encode_mark.pkl')

# Debug: Tampilkan isi booking_status_encode
st.write("Isi booking_status_encode:", booking_status_encode)

st.title("Hotel Booking Status Prediction")

# Fungsi prediksi
def predict_booking_status(no_of_adults, no_of_children, no_of_weekend_nights,
                           no_of_week_nights, type_of_meal_plan, required_car_parking_space,
                           room_type_reserved, lead_time, arrival_year, arrival_month,
                           arrival_date, market_segment_type, repeated_guest,
                           no_of_previous_cancellations, no_of_previous_bookings_not_canceled,
                           avg_price_per_room, no_of_special_requests):

    input_data = pd.DataFrame({
        'no_of_adults': [no_of_adults],
        'no_of_children': [no_of_children],
        'no_of_weekend_nights': [no_of_weekend_nights],
        'no_of_week_nights': [no_of_week_nights],
        'required_car_parking_space': [required_car_parking_space],
        'lead_time': [lead_time],
        'arrival_year': [arrival_year],
        'arrival_month': [arrival_month],
        'arrival_date': [arrival_date],
        'repeated_guest': [repeated_guest],
        'no_of_previous_cancellations': [no_of_previous_cancellations],
        'no_of_previous_bookings_not_canceled': [no_of_previous_bookings_not_canceled],
        'avg_price_per_room': [avg_price_per_room],
        'no_of_special_requests': [no_of_special_requests]
    })

    # One-hot encode fitur kategori
    meal_encoded = oneHot_encode_meal.transform([[type_of_meal_plan]])
    room_encoded = oneHot_encode_room.transform([[room_type_reserved]])
    market_encoded = oneHot_encode_mark.transform([[market_segment_type]])

    # Gabungkan semuanya
    full_input = np.hstack([input_data.values,
                            meal_encoded.toarray() if hasattr(meal_encoded, "toarray") else meal_encoded,
                            room_encoded.toarray() if hasattr(room_encoded, "toarray") else room_encoded,
                            market_encoded.toarray() if hasattr(market_encoded, "toarray") else market_encoded])

    prediction = model.predict(full_input)
    predicted_label = int(prediction[0])  # Pastikan hasil prediksi adalah integer
    st.write("Predicted label (from model):", predicted_label)

    # Map prediksi ke label yang sesuai
    output = {0: "Canceled", 1: "Not_Canceled"}.get(predicted_label, "Unknown")
    return output

# --- Input Manual ---
st.subheader("Manual Input Prediction")

no_of_adults = st.number_input("No of Adults", 0, 100)
no_of_children = st.number_input("No of Children", 0, 100)
no_of_weekend_nights = st.number_input('No of Weekend Night', 0, 2)
no_of_week_nights = st.number_input('No of Week Night', 0, 5)
type_of_meal_plan = st.selectbox('Meal Plan', ['Meal Plan 1', 'Meal Plan 2', 'Meal Plan 3', 'Not Selected'])
required_car_parking_space = st.radio('Required Car Parking Space (0 for No, 1 for Yes)', [0, 1])
room_type_reserved = st.selectbox('Room Type Reserved', ['Room_Type 1', 'Room_Type 2', 'Room_Type 3', 
                                                        'Room_Type 4', 'Room_Type 5', 'Room_Type 6', 'Room_Type 7'])
lead_time = st.number_input("Lead Time (days)", 0, 360)
arrival_year = st.number_input("Arrival Year", 2017, 2018)
arrival_month = st.selectbox('Arrival Month', list(range(1, 13)))
arrival_date = st.selectbox("Arrival Date", list(range(1, 32)))
market_segment_type = st.selectbox('Market Segment Type', ['Offline', 'Online', 'Corporate', 'Aviation', 'Complementary'])
repeated_guest = st.radio('Repeated Guest (0 for No, 1 for Yes)', [0, 1])
no_of_previous_cancellations = st.number_input('Previous Cancellations', 0, 100)
no_of_previous_bookings_not_canceled = st.number_input('Previous Bookings Not Canceled', 0, 100)
avg_price_per_room = st.number_input('Average Price Per Room (in Euros)', 0.00, 10000.00)
no_of_special_requests = st.number_input('Number of Special Requests', 0, 100)

if st.button('Predict from Manual Input'):
    hasil = predict_booking_status(
        no_of_adults, no_of_children, no_of_weekend_nights, no_of_week_nights,
        type_of_meal_plan, required_car_parking_space, room_type_reserved,
        lead_time, arrival_year, arrival_month, arrival_date, market_segment_type,
        repeated_guest, no_of_previous_cancellations, no_of_previous_bookings_not_canceled,
        avg_price_per_room, no_of_special_requests
    )
    st.success(f"Hasil Prediksi: {hasil}")

# --- Test Case 1 ---
st.subheader("Test Case 1")
if st.button("Run Test Case 1"):
    hasil = predict_booking_status(
        no_of_adults=2,
        no_of_children=1,
        no_of_weekend_nights=2,
        no_of_week_nights=3,
        type_of_meal_plan='Meal Plan 1',
        required_car_parking_space=1,
        room_type_reserved='Room_Type 1',
        lead_time=50,
        arrival_year=2018,
        arrival_month=5,
        arrival_date=15,
        market_segment_type='Online',
        repeated_guest=0,
        no_of_previous_cancellations=0,
        no_of_previous_bookings_not_canceled=2,
        avg_price_per_room=120.0,
        no_of_special_requests=1
    )
    st.success(f"Hasil Prediksi Test Case 1: {hasil}")

# --- Test Case 2 ---
st.subheader("Test Case 2")
if st.button("Run Test Case 2"):
    hasil = predict_booking_status(
        no_of_adults=1,
        no_of_children=0,
        no_of_weekend_nights=0,
        no_of_week_nights=1,
        type_of_meal_plan='Not Selected',
        required_car_parking_space=0,
        room_type_reserved='Room_Type 2',
        lead_time=10,
        arrival_year=2018,
        arrival_month=11,
        arrival_date=28,
        market_segment_type='Corporate',
        repeated_guest=1,
        no_of_previous_cancellations=1,
        no_of_previous_bookings_not_canceled=0,
        avg_price_per_room=80.0,
        no_of_special_requests=0
    )
    st.success(f"Hasil Prediksi Test Case 2: {hasil}")



# In[ ]:





# In[ ]:




