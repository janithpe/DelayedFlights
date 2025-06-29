import json
import requests
import streamlit as st

API_URL = "http://127.0.0.1:5000/predict"

st.title("✈️ Flight Delay Predictor")

with st.form("input_form"):
    col1, col2 = st.columns(2)

    with col1:
        day_of_month = st.selectbox("Day of Month", list(range(1, 32)))
        day_of_week = st.selectbox("Day of Week", list(range(1, 8)))
        carrier = st.text_input("Carrier Code (e.g. AA)", value="AA")
        flight_num = st.number_input("Flight Number", value=100)
        tail_num = st.text_input("Tail Number", value="N123AA")

    with col2:
        origin = st.text_input("Origin Airport", value="ORD")
        dest = st.text_input("Destination Airport", value="DFW")
        distance = st.number_input("Distance (miles)", value=1000)
        dep_delay = st.selectbox("Departure Delayed?", options=[0, 1])
        dep_block = st.text_input("Departure Time Block", value="0900-0959")
        arr_block = st.text_input("Arrival Time Block", value="1200-1259")

    origin_seq = st.text_input("Origin Seq ID", value="04")
    dest_seq = st.text_input("Dest Seq ID", value="02")

    submitted = st.form_submit_button("Predict Delay")

if submitted:
    payload = {
        "DAY_OF_MONTH": day_of_month,
        "DAY_OF_WEEK": day_of_week,
        "OP_UNIQUE_CARRIER": carrier,
        "TAIL_NUM": tail_num,
        "OP_CARRIER_FL_NUM": flight_num,
        "ORIGIN": origin,
        "DEST": dest,
        "DEP_DEL15": dep_delay,
        "DEP_TIME_BLK": dep_block,
        "DISTANCE": distance,
        "ARR_TIME_BLK": arr_block,
        "ORIGIN_SEQ_ID": origin_seq,
        "DEST_SEQ_ID": dest_seq
    }

    try:
        response = requests.post(API_URL, json=payload)
        if response.status_code == 200:
            result = response.json()
            st.success(f"Prediction: {'DELAYED' if result['predicted_class'] else 'ON TIME'}")
            st.info(f"Probability of Delay: {result['delay_probability'] * 100:.2f}%")
        else:
            st.error(f"Error from API: {response.text}")
    except Exception as e:
        st.error(f"Request failed: {e}")