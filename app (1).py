
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
from faker import Faker
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Set up fake data generator
fake = Faker()

# Make sure output directory exists in a safe writable location
OUT_DIR = "output"
os.makedirs(OUT_DIR, exist_ok=True)

# Generate fake pharmaceutical supply chain data
def generate_data(n=200):
    data = {
        "Medicine": [fake.word() for _ in range(n)],
        "Supplier": [fake.company() for _ in range(n)],
        "Region": [fake.city() for _ in range(n)],
        "Cost": np.random.randint(10, 100, size=n),
        "Lead_Time_Days": np.random.randint(1, 30, size=n),
        "Quantity": np.random.randint(1, 1000, size=n),
    }
    return pd.DataFrame(data)

# Load or create dataset
df_path = os.path.join(OUT_DIR, "pharma_data.csv")
if os.path.exists(df_path):
    df = pd.read_csv(df_path)
else:
    df = generate_data()
    df.to_csv(df_path, index=False)

# Streamlit dashboard
st.title("Pharmaceutical Supply Chain Dashboard")

# Show dataset
if st.checkbox("Show raw data"):
    st.dataframe(df)

# Plot cost distribution
st.subheader("Cost Distribution")
fig_cost = px.histogram(df, x="Cost", nbins=20, title="Medicine Cost Distribution")
st.plotly_chart(fig_cost)

# Plot lead time by region
st.subheader("Average Lead Time by Region")
lead_time_fig = px.bar(df.groupby("Region")["Lead_Time_Days"].mean().reset_index(),
                       x="Region", y="Lead_Time_Days", title="Lead Time by Region")
st.plotly_chart(lead_time_fig)

# Train LightGBM model
X = df[["Cost", "Lead_Time_Days", "Quantity"]]
y = np.random.randint(50, 500, size=len(df))  # fake target: delivery time
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

params = {
    "objective": "regression",
    "metric": "rmse",
    "verbosity": -1
}

model = lgb.train(params, train_data, valid_sets=[test_data], num_boost_round=50, verbose_eval=False)
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

st.subheader("Model Evaluation")
st.write(f"LightGBM RMSE: {rmse:.2f}")

# Predict for user input
st.subheader("Predict Delivery Time")
cost_input = st.number_input("Cost", min_value=1, max_value=200, value=50)
lead_time_input = st.number_input("Lead Time (days)", min_value=1, max_value=60, value=10)
quantity_input = st.number_input("Quantity", min_value=1, max_value=5000, value=100)

if st.button("Predict"):
    prediction = model.predict(np.array([[cost_input, lead_time_input, quantity_input]]))
    st.success(f"Predicted Delivery Time: {prediction[0]:.2f} hours")
