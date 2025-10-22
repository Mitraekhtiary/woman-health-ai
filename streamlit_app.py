import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

st.title("🌸 Women's Hormonal Health AI Predictor")
st.write("""
This app estimates hormonal balance based on lifestyle, nutrition, and menstrual health. 
It also provides **educational feedback** for improving hormonal health.
""")

# --------------------------
# Step 1: Synthetic training data
# --------------------------
np.random.seed(42)
N = 600

age = np.random.randint(18, 50, N)
bmi = np.random.uniform(18, 32, N)
sleep = np.random.uniform(4, 10, N)
activity = np.random.randint(0, 120, N)
stress = np.random.randint(1, 10, N)
cycle_regular = np.random.randint(0, 2, N)
pms_severity = np.random.randint(1, 10, N)
water_intake = np.random.randint(4, 12, N)
fruit_veg = np.random.randint(0, 8, N)

# target: hormonal_balance
hormonal_balance = (
    80
    - stress * 3
    + sleep * 2
    + activity * 0.1
    - 0.5 * abs(age - 30)
    - 0.5 * abs(bmi - 22)
    + cycle_regular * 5
    - pms_severity * 1.5
    + 0.5 * water_intake
    + 0.5 * fruit_veg
    + np.random.normal(0, 3, N)
)

df_train = pd.DataFrame({
    "age": age,
    "bmi": bmi,
    "sleep": sleep,
    "activity": activity,
    "stress": stress,
    "cycle_regular": cycle_regular,
    "pms_severity": pms_severity,
    "water_intake": water_intake,
    "fruit_veg": fruit_veg,
    "hormonal_balance": hormonal_balance
})

# --------------------------
# Step 2: Train model
# --------------------------
features = ["age","bmi","sleep","activity","stress","cycle_regular","pms_severity","water_intake","fruit_veg"]
target = "hormonal_balance"

model = RandomForestRegressor(n_estimators=150, random_state=42)
model.fit(df_train[features], df_train[target])

# --------------------------
# Step 3: Ask user more questions
# --------------------------
st.subheader("🩺 Lifestyle & Health Inputs")

age_input = st.slider("Age", 15, 60, 25, help="Age can affect hormonal balance")
weight_input = st.number_input("Weight (kg)", 40, 120, 60)
height_input = st.number_input("Height (cm)", 140, 200, 165)
bmi_input = weight_input / ((height_input/100)**2)

sleep_input = st.slider("Average sleep hours per night", 3, 12, 7)
activity_input = st.slider("Daily physical activity (minutes)", 0, 180, 30)
stress_input = st.slider("Stress level (1 = very calm, 10 = very stressed)", 1, 10, 5)

cycle_input = st.selectbox("Is your menstrual cycle regular?", ["Yes", "No"])
cycle_value = 1 if cycle_input=="Yes" else 0
pms_input = st.slider("PMS severity (1 = mild, 10 = severe)", 1, 10, 5)

water_input = st.slider("Daily water intake (cups)", 2, 15, 6)
fruit_veg_input = st.slider("Daily fruit & vegetable servings", 0, 10, 3)

# --------------------------
# Step 4: Prepare user data
# --------------------------
user_df = pd.DataFrame([{
    "age": age_input,
    "bmi": bmi_input,
    "sleep": sleep_input,
    "activity": activity_input,
    "stress": stress_input,
    "cycle_regular": cycle_value,
    "pms_severity": pms_input,
    "water_intake": water_input,
    "fruit_veg": fruit_veg_input
}])

# --------------------------
# Step 5: Prediction
# --------------------------
predicted_score = model.predict(user_df[features])[0]

st.subheader("🔮 Prediction Result")
if predicted_score > 75:
    st.success(f"Score: {predicted_score:.1f} — Excellent hormonal health! 🌿")
elif predicted_score > 55:
    st.warning(f"Score: {predicted_score:.1f} — Moderate. Consider improving sleep, reducing stress, and balanced nutrition.")
else:
    st.error(f"Score: {predicted_score:.1f} — Imbalance detected. Focus on sleep, stress management, and hydration 💧")

# --------------------------
# Step 6: Educational Feedback
# --------------------------
st.subheader("📚 Personalized Feedback")
st.write("- Sleep: 7–9 hours/night helps hormonal balance")
st.write("- Stress: Mindfulness and relaxation reduce cortisol")
st.write("- Physical activity: 30–60 min/day supports metabolism and hormones")
st.write("- Nutrition: Hydration + fruits/vegetables improve hormone regulation")
st.write("- Cycle regularity: Track your cycle to detect irregularities early")
st.write("- PMS: Symptoms reduction improves quality of life")

# --------------------------
# Step 7: Feature Importance Visualization
# --------------------------
importances = model.feature_importances_
importance_df = pd.DataFrame({"feature": features, "importance": importances}).sort_values("importance", ascending=False)

st.subheader("📊 Feature Importance")
st.bar_chart(importance_df.set_index("feature"))
