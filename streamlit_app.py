import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import plotly.graph_objects as go

st.title("🌸 Women's Hormonal Health AI Predictor + Scientific Scoring")
st.write("Personalized scoring, percentile, and actionable recommendations for hormonal health.")

# --------------------------
# Synthetic training data
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

hormonal_balance = (
    80 - stress*3 + sleep*2 + activity*0.1 - 0.5*np.abs(age-30) - 0.5*np.abs(bmi-22)
    + cycle_regular*5 - pms_severity*1.5 + 0.5*water_intake + 0.5*fruit_veg
    + np.random.normal(0,3,N)
)

df_train = pd.DataFrame({
    "age": age, "bmi": bmi, "sleep": sleep, "activity": activity,
    "stress": stress, "cycle_regular": cycle_regular, "pms_severity": pms_severity,
    "water_intake": water_intake, "fruit_veg": fruit_veg, "hormonal_balance": hormonal_balance
})

# --------------------------
# Train model
# --------------------------
features = ["age","bmi","sleep","activity","stress","cycle_regular","pms_severity","water_intake","fruit_veg"]
model = RandomForestRegressor(n_estimators=150, random_state=42)
model.fit(df_train[features], df_train["hormonal_balance"])

# --------------------------
# Healthy ranges
# --------------------------
healthy_ranges = {
    "age": (20,35),
    "bmi": (18.5,24.9),
    "sleep": (7,9),
    "activity": (30,60),
    "stress": (1,4),
    "cycle_regular": (1,1),
    "pms_severity": (1,3),
    "water_intake": (6,10),
    "fruit_veg": (5,8)
}

# --------------------------
# User input
# --------------------------
st.subheader("🩺 Enter Your Health Data")
age_input = st.slider("Age", 15, 60, 25)
weight_input = st.number_input("Weight (kg)", 40, 120, 60)
height_input = st.number_input("Height (cm)", 140, 200, 165)
bmi_input = weight_input / ((height_input/100)**2)
sleep_input = st.slider("Sleep hours/night", 3, 12, 7)
activity_input = st.slider("Physical activity (min/day)", 0, 180, 30)
stress_input = st.slider("Stress level (1=low,10=high)", 1, 10, 5)
cycle_input = st.selectbox("Regular menstrual cycle?", ["Yes","No"])
cycle_value = 1 if cycle_input=="Yes" else 0
pms_input = st.slider("PMS severity (1=mild,10=severe)", 1, 10, 5)
water_input = st.slider("Daily water intake (cups)", 2, 15, 6)
fruit_veg_input = st.slider("Daily fruit & vegetable servings", 0, 10, 3)

user_df = pd.DataFrame([{
    "age": age_input, "bmi": bmi_input, "sleep": sleep_input, "activity": activity_input,
    "stress": stress_input, "cycle_regular": cycle_value, "pms_severity": pms_input,
    "water_intake": water_input, "fruit_veg": fruit_veg_input
}])

# --------------------------
# Prediction
# --------------------------
predicted_score = model.predict(user_df[features])[0]
st.subheader("🔮 Hormonal Balance Prediction")
if predicted_score>75:
    st.success(f"Score: {predicted_score:.1f} — Excellent hormonal health 🌿")
elif predicted_score>55:
    st.warning(f"Score: {predicted_score:.1f} — Moderate health, improve lifestyle")
else:
    st.error(f"Score: {predicted_score:.1f} — Imbalance detected 💧")

# --------------------------
# Step 1: Scientific scoring 0-10 per feature
# --------------------------
st.subheader("📊 Individual Feature Scores")
scores = {}
for feat, val in zip(features, [age_input,bmi_input,sleep_input,activity_input,stress_input,cycle_value,pms_input,water_input,fruit_veg_input]):
    min_h, max_h = healthy_ranges[feat]
    if feat in ["stress","pms_severity"]:  # lower is better
        score = max(0, 10*(max_h-val)/(max_h-min_h))
    else:  # higher within range is better
        if val<min_h:
            score = max(0, 10*val/min_h)
        elif val>max_h:
            score = max(0, 10*(max_h/val))
        else:
            score = 10
    scores[feat] = round(score,1)
st.write(pd.DataFrame(list(scores.items()), columns=["Feature","Score (0-10)"]))

# --------------------------
# Step 2: Overall percentile
# --------------------------
overall_percentile = np.mean(list(scores.values()))*10  # scale 0-100
st.subheader("📈 Overall Hormonal Health Percentile")
st.progress(int(overall_percentile))
st.write(f"Your overall hormonal health percentile: **{overall_percentile:.1f}/100**")

# --------------------------
# Step 3: Top 3 areas to improve
# --------------------------
st.subheader("⚡ Top 3 Areas to Improve")
sorted_feats = sorted(scores.items(), key=lambda x: x[1])
for feat, score in sorted_feats[:3]:
    st.write(f"- **{feat}**: score {score}/10 — consider adjusting towards healthy range")

# --------------------------
# Step 4: Radar chart
# --------------------------
st.subheader("📊 Radar Chart Comparison")
fig = go.Figure()
fig.add_trace(go.Scatterpolar(r=list(scores.values()), theta=list(features),
                              fill='toself', name='Your Scores', line_color='crimson'))
fig.add_trace(go.Scatterpolar(r=[10]*len(features), theta=list(features),
                              fill='toself', name='Ideal Score', line_color='green'))
fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,10])), showlegend=True)
st.plotly_chart(fig)
