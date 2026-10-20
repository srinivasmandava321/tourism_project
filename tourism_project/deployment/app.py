import os
import streamlit as st
import pandas as pd
import joblib

# Load the model committed by the pipeline (sits next to this file)
model_path = os.path.join(os.path.dirname(__file__), "best_tourism_project_model_v1.joblib")
model = joblib.load(model_path)

st.title("Tourism Package Purchase Prediction App")
st.write("""
This application predicts whether a customer is likely to purchase a tourism package.
Enter the customer profile, travel preferences, and sales interaction details below
to get a prediction.
""")

age = st.number_input("Age", 18, 100, 35)
contact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
city_tier = st.selectbox("City Tier", [1, 2, 3])
duration = st.number_input("Duration of Pitch (minutes)", 0.0, 300.0, 15.0, 1.0)
occupation = st.selectbox(
    "Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"]
)
gender = st.selectbox("Gender", ["Female", "Male"])
persons = st.number_input("Number of People Visiting", 1, 20, 3)
followups = st.number_input("Number of Follow-ups", 0.0, 50.0, 3.0, 1.0)
product = st.selectbox(
    "Product Pitched", ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"]
)
property_star = st.selectbox("Preferred Property Star Rating", [3, 4, 5])
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
trips = st.number_input("Number of Trips", 0.0, 100.0, 2.0, 1.0)
passport = st.selectbox("Passport", [0, 1])
satisfaction = st.selectbox("Pitch Satisfaction Score", [1, 2, 3, 4, 5])
own_car = st.selectbox("Own Car", [0, 1])
children = st.number_input("Number of Children Visiting", 0.0, 19.0, 1.0, 1.0)
designation = st.selectbox(
    "Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"]
)
income = st.number_input("Monthly Income", 0.0, 1000000.0, 25000.0, 500.0)

input_data = pd.DataFrame([{
    "Age": age,
    "TypeofContact": contact,
    "CityTier": city_tier,
    "DurationOfPitch": duration,
    "Occupation": occupation,
    "Gender": gender,
    "NumberOfPersonVisiting": persons,
    "NumberOfFollowups": followups,
    "ProductPitched": product,
    "PreferredPropertyStar": property_star,
    "MaritalStatus": marital_status,
    "NumberOfTrips": trips,
    "Passport": passport,
    "PitchSatisfactionScore": satisfaction,
    "OwnCar": own_car,
    "NumberOfChildrenVisiting": children,
    "Designation": designation,
    "MonthlyIncome": income,
}])

if st.button("Predict Purchase"):
    prediction = model.predict(input_data)[0]
    result = "Likely to Purchase" if prediction == 1 else "Unlikely to Purchase"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts Customer will : **{result}**")
