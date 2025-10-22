import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

st.title("🌸 Women's Hormonal Balance Predictor")
st.write("AI-powered estimation of hormonal balance based on lifestyle inputs.")

# --------------------------
# Step 1: Generate synthetic training data
# --------------------------
np.random.seed(42)
N = 500  # number of synthetic samples

age = np.random.randint(18, 50, N)
stress = np.random.randint(1, 11, N)
sleep = np.random.uniform(4, 10, N)
activity = np.random.randint(0, 120, N)
cycle_regular = np.random.randint(0, 2, N)

# target: hormonal_balance
hormonal_balance = (
    80 
    - stress * 3
    + sleep * 2
    + activity * 0.1
    - 0.5 * np.abs(age - 30)
    + cycle_regular * 5
    + np.random.normal(0, 3, N)  # add some noise
)

df_train = pd.DataFrame({
    "age": age,
    "stress": stress,
    "sleep": sleep,
    "activity": activity,
    "cycle_regular": cycle_regular,
    "hormonal_balance": hormonal_balance
})

# --------------------------
# Step 2: Train model
# --------------------------
features = ["age", "stress", "sleep", "activity", "cycle_regular"]
target = "hormonal_balance"

model = RandomForestRegressor(n_estimators=120, random_state=42)
model.fit(df_train[features], df_train[target])

# --------------------------
# Step 3: User input
# --------------------------
st.subheader("🩺 Please answer the following questions:")

age_input = st.slider("Age", 15, 60, 25)
stress_input = st.slider("Stress level (1 = very calm, 10 = very stressed)", 1, 10, 5)
sleep_input = st.slider("Average sleep hours per night", 3.0, 12.0, 7.0)
activity_input = st.slider("Daily physical activity (minutes)", 0, 180, 30)
cycle_input = st.selectbox("Is your menstrual cycle regular?", ["Yes", "No"])
cycle_value = 1 if cycle_input == "Yes" else 0

user_df = pd.DataFrame([{
    "age": age_input,
    "stress": stress_input,
    "sleep": sleep_input,
    "activity": activity_input,
    "cycle_regular": cycle_value
}])

# --------------------------
# Step 4: Prediction
# --------------------------
predicted_score = model.predict(user_df[features])[0]

st.subheader("🔮 Prediction Result:")
if predicted_score > 70:
    st.success(f"Your estimated hormonal balance score is {predicted_score:.1f} — looks healthy! 🌿")
elif predicted_score > 50:
    st.warning(f"Your balance score is {predicted_score:.1f} — moderate, consider more sleep or less stress.")
else:
    st.error(f"Your balance score is {predicted_score:.1f} — signs of imbalance detected, try to rest more 💧")

# --------------------------
# Optional: Feature impact
# --------------------------
importances = model.feature_importances_
importance_df = pd.DataFrame({"feature": features, "importance": importances}).sort_values(by="importance", ascending=False)

st.subheader("📊 Feature Importance")
st.bar_chart(importance_df.set_index("feature"))

st.markdown("""
### 💡 Insights:
- **Stress** and **sleep quality** are the strongest predictors.
- Maintaining **physical activity** and **regular menstrual cycles** improves balance.
""")
