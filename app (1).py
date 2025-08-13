import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
from faker import Faker
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta

OUT_DIR = "output"
os.makedirs(OUT_DIR, exist_ok=True)
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
fake = Faker('en_US')

@st.cache_data
def generate_pharma_data(n_rows=5000):
    product_categories = ['Antibiotics', 'Pain Relievers', 'Vitamins', 'Cardiovascular']
    regions = ['Tehran', 'Mashhad', 'Isfahan', 'Shiraz', 'Tabriz']
    
    data = {
        "Product_ID": [f'PROD_{i:04d}' for i in range(n_rows)],
        "Medicine": [fake.word().capitalize() + ' ' + str(np.random.randint(50, 1000)) + 'mg' for _ in range(n_rows)],
        "Category": np.random.choice(product_categories, size=n_rows),
        "Supplier": [fake.company() for _ in range(n_rows)],
        "Region": np.random.choice(regions, size=n_rows),
        "Unit_Cost": np.random.lognormal(mean=np.log(25), sigma=0.8, size=n_rows).round(2),
        "Quantity_Shipped": np.random.randint(100, 5000, size=n_rows),
        "Lead_Time_Days": np.random.randint(1, 30, size=n_rows),
        "Ship_Date": [fake.date_between(start_date='-2y', end_date='today') for _ in range(n_rows)],
    }
    df = pd.DataFrame(data)
    df['Total_Cost'] = df['Unit_Cost'] * df['Quantity_Shipped']
    return df

df_path = os.path.join(OUT_DIR, "pharma_data_comprehensive.csv")
if os.path.exists(df_path):
    df = pd.read_csv(df_path)
    df['Ship_Date'] = pd.to_datetime(df['Ship_Date'])
else:
    df = generate_pharma_data()
    df.to_csv(df_path, index=False)

st.set_page_config(layout="wide")
st.title("💊 Pharmaceutical Supply Chain Dashboard")
st.markdown("This dashboard helps to analyze and predict supply chain data.")

st.sidebar.header("Filter Data")
selected_regions = st.sidebar.multiselect(
    "Select Region",
    options=df['Region'].unique(),
    default=df['Region'].unique()
)

min_cost, max_cost = float(df['Total_Cost'].min()), float(df['Total_Cost'].max())
cost_range = st.sidebar.slider(
    "Select Total Cost Range",
    min_value=min_cost,
    max_value=max_cost,
    value=(min_cost, max_cost)
)

filtered_df = df[
    (df['Region'].isin(selected_regions)) &
    (df['Total_Cost'] >= cost_range[0]) &
    (df['Total_Cost'] <= cost_range[1])
]

st.info(f"Showing **{len(filtered_df)}** out of **{len(df)}** records.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Total Cost by Supplier")
    supplier_costs = filtered_df.groupby("Supplier")['Total_Cost'].sum().nlargest(10).reset_index()
    fig_supplier = px.bar(supplier_costs, x="Supplier", y="Total_Cost", 
                          title="Top 10 Suppliers by Total Cost")
    st.plotly_chart(fig_supplier, use_container_width=True)

with col2:
    st.subheader("Quantity Shipped by Category")
    category_quantity = filtered_df.groupby("Category")['Quantity_Shipped'].sum().reset_index()
    fig_category = px.pie(category_quantity, values="Quantity_Shipped", names="Category", 
                          title="Quantity Shipped by Category")
    st.plotly_chart(fig_category, use_container_width=True)

st.subheader("Total Cost Over Time")
time_series_data = filtered_df.groupby(pd.Grouper(key='Ship_Date', freq='M'))['Total_Cost'].sum().reset_index()
fig_time_series = px.line(time_series_data, x="Ship_Date", y="Total_Cost", 
                           title="Monthly Total Cost Trend")
st.plotly_chart(fig_time_series, use_container_width=True)

with st.expander("Show Raw Data"):
    st.dataframe(filtered_df)

st.markdown("---")
st.header("🔬 Lead Time Prediction Model")
st.markdown("This model predicts lead time based on costs and quantities.")

@st.cache_resource
def train_model():
    X = df[['Total_Cost', 'Quantity_Shipped', 'Unit_Cost']]
    y = df['Lead_Time_Days']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)
    
    params = {
        "objective": "regression",
        "metric": "rmse",
        "verbosity": -1,
        "n_estimators": 100
    }
    
    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    
    return model, rmse, X_test, y_test, predictions

model, rmse, X_test, y_test, predictions = train_model()

st.subheader("Model Evaluation")
st.write(f"**Root Mean Squared Error (RMSE):** `{rmse:.2f}` days")

eval_df = pd.DataFrame({'Actual Lead Time': y_test, 'Predicted Lead Time': predictions})
eval_fig = px.scatter(eval_df, x='Actual Lead Time', y='Predicted Lead Time', 
                      title='Actual vs. Predicted Lead Time')
st.plotly_chart(eval_fig, use_container_width=True)

st.subheader("Predict Lead Time for a New Order")
col3, col4, col5 = st.columns(3)
with col3:
    unit_cost_input = st.number_input("Unit Cost", min_value=1.0, max_value=10000.0, value=250.0)
with col4:
    quantity_input = st.number_input("Quantity", min_value=1, max_value=10000, value=500)
with col5:
    total_cost_input = st.number_input("Total Cost", min_value=1.0, max_value=1000000.0, value=125000.0)

if st.button("Predict Lead Time"):
    new_data = np.array([[total_cost_input, quantity_input, unit_cost_input]])
    prediction = model.predict(new_data)
    st.success(f"Predicted Lead Time: **{prediction[0]:.2f}** days")
