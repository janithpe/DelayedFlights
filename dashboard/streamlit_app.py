import os
import json
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:5000/predict")

st.markdown("<h1 style='text-align: center; color: black;'>✈️ Flight Delay Predictor</h1>", unsafe_allow_html=True)

# Create a dictionaries to map form inputs
dep_delay_map = {"No":0, "Yes":1}
day_names_map = {"Monday":1, "Tuesday":2, "Wednesday":3, "Thursday":4, "Friday":5, "Saturday":6, "Sunday":7}

# Create a list of options to time block inputs
time_blocks_list = ["0000-0059", "0100-0159", "0200-0259", "0300-0359", "0400-0459", "0500-0559",
                    "0600-0659", "0700-0759", "0800-0859", "0900-0959", "1000-1059", "1100-1159",
                    "1200-1259", "1300-1359", "1400-1459", "1500-1559", "1600-1659", "1700-1759",
                    "1800-1859", "1900-1959", "2000-2059", "2100-2159", "2200-2259", "2300-2359"]

with st.form("input_form"):
    col1, col2 = st.columns(2)

    with col1:
        dep_delay = st.selectbox("Departure Delayed?", options=list(dep_delay_map.keys()))
        flight_num = st.number_input("Flight Number", value=1533)
        tail_num = st.text_input("Tail Number", value="N123AA")
        distance = st.number_input("Distance (miles)", value=1013)
        dep_block = st.selectbox("Departure Time Block", options=list(time_blocks_list))
        origin_seq = st.text_input("Origin Seq ID", value="04")        

    with col2:
        day_of_month = st.selectbox("Day of Month", list(range(1, 32)))
        day_of_week = st.selectbox("Day of Week", options=list(day_names_map.keys()))
        origin = st.text_input("Origin Airport", value="ORD")
        dest = st.text_input("Destination Airport", value="DFW")
        arr_block = st.selectbox("Arrival Time Block", options=list(time_blocks_list))
        dest_seq = st.text_input("Dest Seq ID", value="02")

    carrier = st.text_input("Carrier Code (e.g. AA)", value="AA")
    
    submitted = st.form_submit_button("Predict Delay")

if submitted:
    payload = {
        "DAY_OF_MONTH": day_of_month,
        "DAY_OF_WEEK": day_names_map[day_of_week],
        "OP_UNIQUE_CARRIER": carrier,
        "TAIL_NUM": tail_num,
        "OP_CARRIER_FL_NUM": flight_num,
        "ORIGIN": origin,
        "DEST": dest,
        "DEP_DEL15": dep_delay_map[dep_delay],
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
            if result['predicted_class']:
                # Display "DELAYED" with a red background
                st.markdown(
                    """
                    <style>
                    .delayed-prediction {
                        background-color: rgba(255, 0, 0, 0.2);
                        color: red;
                        padding: 15px;
                        border-radius: 5px;
                    }
                    </style>
                    <p class="delayed-prediction">Prediction: DELAYED</p>
                    """,
                    unsafe_allow_html=True
                )
            else:
                # Display "ON TIME" with a green background
                st.markdown(
                    """
                    <style>
                    .ontime-prediction {
                    background-color: rgba(0, 255, 0, 0.2);
                    color: green;
                    padding: 15px;
                    border-radius: 5px;
                    }
                    </style>
                    <p class="ontime-prediction">Prediction: ON TIME</p>
                    """,
                    unsafe_allow_html=True
                )
            st.info(f"Probability of Delay: {result['delay_probability'] * 100:.2f}%")
        else:
            st.error(f"Error from API: {response.text}")
    except Exception as e:
        st.error(f"Request failed: {e}")