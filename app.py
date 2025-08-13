
import os
import math
import warnings
from collections import defaultdict, deque
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from faker import Faker
import random

import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

# ------------------ CONFIG ------------------
OUT_DIR = '/mnt/data/outputs'
os.makedirs(OUT_DIR, exist_ok=True)

RANDOM_SEED = 1234
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
fake = Faker('en_US')
Faker.seed(RANDOM_SEED)

# Defaults (these are configurable via sidebar)
DEFAULTS = {
    'NUM_PRODUCTS': 500,
    'NUM_WAREHOUSES': 6,
    'NUM_SUPPLIERS': 30,
    'NUM_CUSTOMERS': 400,
    'NUM_PURCHASE_ORDERS': 2000,
    'NUM_SALES_RECORDS': 8000,
    'NUM_RETURNS_RATIO': 0.04,
    'SIM_YEARS': 1.5
}

SERVICE_LEVEL = 0.95
ORDERING_COST = 100
HOLDING_COST_RATE = 0.20

# ------------------ HELPERS ------------------
@st.cache_data
def get_z_from_service_level(service_level: float) -> float:
    from scipy.stats import norm
    return float(norm.ppf(service_level))

def get_seasonal_factor(date: datetime) -> float:
    month = date.month
    if month in [12, 1, 2]:
        return 1.5
    elif month in [7, 8]:
        return 0.8
    return 1.0

# ------------------ DATA GENERATORS ------------------
def generate_products_df(num_products: int) -> pd.DataFrame:
    product_ids = [f'PROD_{i:05d}' for i in range(1, num_products + 1)]
    product_categories = [
        'Antibiotics', 'Pain Relievers', 'Anti-Inflammatories', 'Vitamins & Supplements',
        'Cardiovascular', 'Gastrointestinal', 'Respiratory', 'Neurological',
        'Dermatological', 'Ophthalmic & Otic', 'Diabetic Care', 'Hormonal',
        'Herbal Remedies', 'Disposable Medical Devices'
    ]

    product_names = [
        f'{random.choice(["Tablet", "Capsule", "Syrup", "Ointment", "Ampoule", "Drop", "Cream"]) } {fake.word().capitalize()} {fake.word().capitalize()}'
        if random.random() < 0.75
        else f'{"".join(random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ", k=2))}-{random.randint(100,999)} {random.choice(["mg", "g", "ml", "unit"])} {fake.word().capitalize()}'
        for _ in range(num_products)
    ]

    unit_prices = np.round(np.random.lognormal(mean=np.log(20), sigma=0.9, size=num_products), 2)
    unit_prices = np.maximum(unit_prices, 0.5)

    shelf_life_days = np.random.choice([365, 730, 1095, 1460, 1825], size=num_products, p=[0.1, 0.4, 0.3, 0.1, 0.1])
    storage_conditions = np.random.choice(['Room Temperature', 'Cold Storage', 'Protect from Light/Moisture', 'Refrigerated (2-8°C)'], size=num_products, p=[0.6, 0.2, 0.1, 0.1])

    products_df = pd.DataFrame({
        'product_id': product_ids,
        'product_name': product_names,
        'product_category': np.random.choice(product_categories, size=num_products),
        'unit_price': unit_prices,
        'avg_shelf_life_days': shelf_life_days,
        'storage_conditions': storage_conditions
    })
    products_df['product_category'] = products_df['product_category'].astype('category')
    products_df['storage_conditions'] = products_df['storage_conditions'].astype('category')
    return products_df

def generate_warehouses_df(num_warehouses: int) -> pd.DataFrame:
    warehouse_locations = {
        'Tehran': {'lat': 35.6892, 'lon': 51.3890}, 'Mashhad': {'lat': 36.2605, 'lon': 59.6168},
        'Isfahan': {'lat': 32.6546, 'lon': 51.6670}, 'Tabriz': {'lat': 38.0805, 'lon': 46.2918},
        'Shiraz': {'lat': 29.6065, 'lon': 52.5414}, 'Ahvaz': {'lat': 31.3204, 'lon': 48.6720},
        'Karaj': {'lat': 35.8322, 'lon': 50.9667}, 'Rasht': {'lat': 37.2752, 'lon': 49.5891},
        'Kerman': {'lat': 30.2832, 'lon': 57.0671}, 'Bandar Abbas': {'lat': 27.1887, 'lon': 56.2829}
    }
    cities = list(warehouse_locations.keys())[:num_warehouses]
    warehouse_ids = [f'WH_{i:02d}' for i in range(1, num_warehouses + 1)]
    warehouse_names = [f'Central Warehouse {city}' for city in cities]
    warehouse_latitudes = [warehouse_locations[c]['lat'] for c in cities]
    warehouse_longitudes = [warehouse_locations[c]['lon'] for c in cities]
    capacity_units = np.random.randint(1_000_000, 5_000_000, size=num_warehouses)
    if num_warehouses > 0:
        capacity_units[0] = np.random.randint(4_000_000, 7_000_000)

    warehouses_df = pd.DataFrame({
        'warehouse_id': warehouse_ids,
        'warehouse_name': warehouse_names,
        'warehouse_city': cities,
        'latitude': warehouse_latitudes,
        'longitude': warehouse_longitudes,
        'capacity_units': capacity_units
    })
    warehouses_df['warehouse_city'] = warehouses_df['warehouse_city'].astype('category')
    return warehouses_df

def generate_suppliers_df(num_suppliers: int) -> pd.DataFrame:
    supplier_ids = [f'SUP_{i:03d}' for i in range(1, num_suppliers + 1)]
    supplier_names = [fake.company() for _ in range(num_suppliers)]
    supplier_locations = [fake.city() for _ in range(num_suppliers)]
    avg_lead_time_days = np.random.randint(5, 30, size=num_suppliers)
    reliability_scores = np.round(np.random.uniform(0.7, 0.99, size=num_suppliers), 2)

    suppliers_df = pd.DataFrame({
        'supplier_id': supplier_ids,
        'supplier_name': supplier_names,
        'supplier_location': supplier_locations,
        'avg_lead_time_days': avg_lead_time_days,
        'reliability_score': reliability_scores
    })
    suppliers_df['supplier_location'] = suppliers_df['supplier_location'].astype('category')
    return suppliers_df

def generate_purchase_orders_df(num_purchase_orders: int, products_df: pd.DataFrame, suppliers_df: pd.DataFrame, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    po_data = []
    days_range = (end_date - start_date).days
    supplier_weights = suppliers_df['reliability_score'].values

    for i in range(num_purchase_orders):
        po_id = f'PO_{i:06d}'
        po_date = start_date + timedelta(days=random.randint(0, days_range))

        product = products_df.sample(1).iloc[0]
        supplier = suppliers_df.sample(1, weights=supplier_weights).iloc[0]

        base_quantity = np.random.randint(500, 5000)
        ordered_quantity = max(10, int(base_quantity / (product['unit_price'] / 10 + 1)))
        unit_cost = max(0.1, product['unit_price'] * np.random.uniform(0.5, 0.85))
        expected_delivery_date = po_date + timedelta(days=int(supplier['avg_lead_time_days']))

        delivery_delay_days = 0
        if random.random() < (1 - supplier['reliability_score']):
            delivery_delay_days = np.random.randint(1, 10)
            if random.random() < 0.2:
                delivery_delay_days = np.random.randint(10, 30)

        actual_delivery_date = expected_delivery_date + timedelta(days=int(delivery_delay_days))

        delivery_status = 'Delivered'
        if actual_delivery_date > end_date:
            delivery_status = 'Pending'
        elif actual_delivery_date > expected_delivery_date:
            delivery_status = 'Delayed'

        po_data.append({
            'po_id': po_id,
            'po_date': po_date,
            'supplier_id': supplier['supplier_id'],
            'product_id': product['product_id'],
            'ordered_quantity': ordered_quantity,
            'unit_cost': unit_cost,
            'expected_delivery_date': expected_delivery_date,
            'actual_delivery_date': actual_delivery_date,
            'delivery_status': delivery_status
        })

    purchase_orders_df = pd.DataFrame(po_data)
    for c in ['po_date', 'expected_delivery_date', 'actual_delivery_date']:
        purchase_orders_df[c] = pd.to_datetime(purchase_orders_df[c])
    purchase_orders_df['delivery_status'] = purchase_orders_df['delivery_status'].astype('category')
    return purchase_orders_df

def generate_customers_df(num_customers: int) -> pd.DataFrame:
    customer_ids = [f'CUST_{i:05d}' for i in range(1, num_customers + 1)]
    customer_names = [fake.company() + ' Pharmacy' for _ in range(num_customers)]
    customer_locations = [fake.city() for _ in range(num_customers)]

    customers_df = pd.DataFrame({
        'customer_id': customer_ids,
        'customer_name': customer_names,
        'customer_location': customer_locations
    })
    customers_df['customer_location'] = customers_df['customer_location'].astype('category')
    return customers_df

def generate_sales_df(num_sales_records: int, products_df: pd.DataFrame, customers_df: pd.DataFrame, warehouses_df: pd.DataFrame, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    sales_data = []
    days_range = (end_date - start_date).days

    for i in range(num_sales_records):
        sales_id = f'SALE_{i:07d}'
        sales_date = start_date + timedelta(days=random.randint(0, days_range))

        product = products_df.sample(1).iloc[0]
        customer = customers_df.sample(1).iloc[0]
        warehouse = warehouses_df.sample(1).iloc[0]

        base_sold_quantity = np.random.randint(1, 120)
        adjusted_sold_quantity = max(1, int(base_sold_quantity / (product['unit_price'] / 10 + 1)))

        seasonal_factor = get_seasonal_factor(sales_date)
        sold_quantity = max(1, int(adjusted_sold_quantity * seasonal_factor * np.random.uniform(0.7, 1.3)))

        sales_price = round(product['unit_price'] * sold_quantity, 2)
        delivery_time_hours = np.random.randint(4, 72)

        sales_data.append({
            'sales_id': sales_id,
            'sales_date': sales_date,
            'customer_id': customer['customer_id'],
            'warehouse_id': warehouse['warehouse_id'],
            'product_id': product['product_id'],
            'sold_quantity': sold_quantity,
            'sales_price': sales_price,
            'delivery_time_hours': delivery_time_hours
        })

    sales_df = pd.DataFrame(sales_data)
    sales_df['sales_date'] = pd.to_datetime(sales_df['sales_date'])
    return sales_df

def generate_returns_df(sales_df: pd.DataFrame, num_returns: int) -> pd.DataFrame:
    if sales_df.empty or num_returns == 0:
        return pd.DataFrame(columns=['return_id', 'sales_id', 'return_date', 'product_id', 'warehouse_id', 'returned_quantity', 'return_reason'])

    return_reasons = ['Damaged in transit', 'Incorrect item received', 'Customer changed mind', 'Expired product received']
    returns_data = []

    returned_sales = sales_df.sample(n=num_returns, replace=True).reset_index(drop=True)

    for index, row in returned_sales.iterrows():
        return_date = row['sales_date'] + timedelta(days=random.randint(1, 14))
        returned_quantity = random.randint(1, max(1, int(row['sold_quantity'])))
        return_reason = random.choice(return_reasons)

        returns_data.append({
            'return_id': f'RET_{index:06d}',
            'sales_id': row['sales_id'],
            'return_date': return_date,
            'product_id': row['product_id'],
            'warehouse_id': row['warehouse_id'],
            'returned_quantity': returned_quantity,
            'return_reason': return_reason
        })

    returns_df = pd.DataFrame(returns_data)
    returns_df['return_date'] = pd.to_datetime(returns_df['return_date'])
    returns_df['return_reason'] = returns_df['return_reason'].astype('category')
    return returns_df

# ------------------ INVENTORY SIMULATION ------------------
def simulate_inventory_and_waste(purchase_orders_df: pd.DataFrame, sales_df: pd.DataFrame, returns_df: pd.DataFrame, products_df: pd.DataFrame, warehouses_df: pd.DataFrame):
    receipts_df = purchase_orders_df[purchase_orders_df['delivery_status'] == 'Delivered'].copy()
    if not receipts_df.empty:
        receipts_df['event_date'] = receipts_df['actual_delivery_date']
        receipts_df['quantity_change'] = receipts_df['ordered_quantity']
        receipts_df['event_type'] = 'Receipt'
        receipts_df['batch_number'] = 'BATCH_' + receipts_df['po_id']
        receipts_df = receipts_df.merge(products_df[['product_id', 'avg_shelf_life_days']], on='product_id', how='left')
        receipts_df['expiry_date'] = receipts_df['event_date'] + pd.to_timedelta(receipts_df['avg_shelf_life_days'], unit='D')
        receipts_df['warehouse_id'] = np.random.choice(warehouses_df['warehouse_id'], size=len(receipts_df))
    else:
        receipts_df = pd.DataFrame(columns=['event_date','product_id','warehouse_id','quantity_change','event_type','batch_number','expiry_date'])

    issues_df = sales_df.copy()
    issues_df['event_date'] = issues_df['sales_date']
    issues_df['quantity_change'] = -issues_df['sold_quantity']
    issues_df['event_type'] = 'Issue'
    issues_df['batch_number'] = None
    issues_df['expiry_date'] = None

    if not returns_df.empty:
        returns_events_df = returns_df.copy()
        returns_events_df['event_date'] = returns_events_df['return_date']
        returns_events_df['quantity_change'] = returns_events_df['returned_quantity']
        returns_events_df['event_type'] = 'Return_Receipt'
        returns_events_df['batch_number'] = 'RET_BATCH_' + returns_events_df['return_id']
        returns_events_df = returns_events_df.merge(products_df[['product_id', 'avg_shelf_life_days']], on='product_id', how='left')
        returns_events_df['expiry_date'] = returns_events_df['event_date'] + pd.to_timedelta(returns_events_df['avg_shelf_life_days'], unit='D')
    else:
        returns_events_df = pd.DataFrame(columns=['event_date','product_id','warehouse_id','quantity_change','event_type','batch_number','expiry_date'])

    combined = pd.concat([
        receipts_df[['event_date','product_id','warehouse_id','quantity_change','event_type','batch_number','expiry_date']],
        issues_df[['event_date','product_id','warehouse_id','quantity_change','event_type','batch_number','expiry_date']],
        returns_events_df[['event_date','product_id','warehouse_id','quantity_change','event_type','batch_number','expiry_date']]
    ], axis=0, ignore_index=True)

    if combined.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    combined['event_date'] = pd.to_datetime(combined['event_date']).dt.normalize()
    combined = combined.sort_values('event_date').reset_index(drop=True)

    start_day = combined['event_date'].min().date()
    end_day = combined['event_date'].max().date()
    all_days = pd.date_range(start_day, end_day, freq='D')

    inventory = defaultdict(deque)
    waste_records = []
    stockout_records = []

    events_by_day = {d.date(): df for d, df in combined.groupby(combined['event_date'])}

    for current in all_days:
        day = current.date()
        # expiries
        keys_to_check = list(inventory.keys())
        for key in keys_to_check:
            wh, prod = key
            dq = inventory[key]
            while dq and dq[0][0].date() <= day:
                expiry_date, batch_id, qty = dq.popleft()
                if qty > 0:
                    waste_records.append({
                        'waste_date': pd.Timestamp(day),
                        'warehouse_id': wh,
                        'product_id': prod,
                        'batch_number': batch_id,
                        'wasted_quantity': qty,
                        'reason': 'Expired'
                    })
        # events
        if day in events_by_day:
            df_day = events_by_day[day]
            for ev in df_day.itertuples(index=False):
                wh = ev.warehouse_id
                prod = ev.product_id
                key = (wh, prod)
                if ev.event_type in ['Receipt', 'Return_Receipt']:
                    expiry = ev.expiry_date if not pd.isna(ev.expiry_date) else pd.Timestamp(day) + pd.Timedelta(days=365)
                    inventory[key].append((pd.to_datetime(expiry), ev.batch_number, int(ev.quantity_change)))
                    inventory[key] = deque(sorted(inventory[key], key=lambda x: x[0]))
                elif ev.event_type == 'Issue':
                    qty_needed = int(abs(ev.quantity_change))
                    if key not in inventory or sum(b[2] for b in inventory[key]) == 0:
                        stockout_records.append({
                            'stockout_date': pd.Timestamp(day),
                            'warehouse_id': wh,
                            'product_id': prod,
                            'stockout_quantity': qty_needed
                        })
                        continue
                    available = sum(b[2] for b in inventory[key])
                    if qty_needed > available:
                        stockout_records.append({
                            'stockout_date': pd.Timestamp(day),
                            'warehouse_id': wh,
                            'product_id': prod,
                            'stockout_quantity': qty_needed - available
                        })
                        qty_to_take = available
                    else:
                        qty_to_take = qty_needed

                    while qty_to_take > 0 and inventory[key]:
                        expiry_date, batch_id, q = inventory[key].popleft()
                        take = min(q, qty_to_take)
                        q_remain = q - take
                        qty_to_take -= take
                        if q_remain > 0:
                            inventory[key].appendleft((expiry_date, batch_id, q_remain))

    waste_df = pd.DataFrame(waste_records)
    if not waste_df.empty:
        waste_df['waste_date'] = pd.to_datetime(waste_df['waste_date'])
        waste_df['reason'] = waste_df['reason'].astype('category')

    stockout_df = pd.DataFrame(stockout_records)
    if not stockout_df.empty:
        stockout_df['stockout_date'] = pd.to_datetime(stockout_df['stockout_date'])

    inventory_snapshot = []
    for (wh, prod), dq in inventory.items():
        total_qty = sum(b[2] for b in dq)
        next_expiry = dq[0][0] if dq else pd.NaT
        inventory_snapshot.append({'warehouse_id': wh, 'product_id': prod, 'on_hand_qty': total_qty, 'next_expiry': next_expiry})
    inventory_snapshot_df = pd.DataFrame(inventory_snapshot)

    return inventory_snapshot_df, waste_df, stockout_df

# ------------------ EDA ------------------
def perform_eda(products_df, sales_df, waste_df, stockout_df):
    sales_by_date = sales_df.groupby(sales_df['sales_date'].dt.date)['sold_quantity'].sum()
    sales_ts = sales_by_date.rename_axis('date').reset_index(name='sold_quantity')
    # plotly timeseries
    fig = px.line(sales_ts, x='date', y='sold_quantity', title='Total Sold Quantity Over Time')
    return fig

# ------------------ FORECAST ------------------
def train_forecast_for_product(sales_df, product_id, n_splits=3):
    product_sales = sales_df[sales_df['product_id'] == product_id].copy()
    if product_sales.empty:
        raise ValueError('No sales for this product')

    daily = product_sales.groupby(product_sales['sales_date'].dt.normalize())['sold_quantity'].sum().rename('y').reset_index()
    daily = daily.set_index('sales_date').resample('D').sum().fillna(0).reset_index().rename(columns={'sales_date':'ds'})

    daily['year'] = daily['ds'].dt.year
    daily['month'] = daily['ds'].dt.month
    daily['dayofweek'] = daily['ds'].dt.dayofweek
    daily['lag_1'] = daily['y'].shift(1).fillna(0)
    daily['lag_7'] = daily['y'].shift(7).fillna(0)

    feature_cols = ['year','month','dayofweek','lag_1','lag_7']
    X = daily[feature_cols]
    y = daily['y']

    tscv = TimeSeriesSplit(n_splits=n_splits)
    maes = []
    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        model = lgb.LGBMRegressor(random_state=RANDOM_SEED, n_estimators=300)
        model.fit(X_train, y_train, eval_set=[(X_val,y_val)], early_stopping_rounds=30, verbose=False)
        preds = model.predict(X_val)
        maes.append(mean_absolute_error(y_val, preds))

    final_model = lgb.LGBMRegressor(random_state=RANDOM_SEED, n_estimators=300)
    final_model.fit(X, y)

    last_train_idx, last_val_idx = list(tscv.split(X))[-1]
    X_test = X.iloc[last_val_idx]
    y_test = y.iloc[last_val_idx]
    preds_test = final_model.predict(X_test)
    final_mae = mean_absolute_error(y_test, preds_test)

    fi = pd.DataFrame({'feature': feature_cols, 'importance': final_model.feature_importances_}).sort_values('importance', ascending=False)
    fi.to_csv(os.path.join(OUT_DIR, f'feature_importance_{product_id}.csv'), index=False)

    # prepare plotly figure for actual vs pred (last fold)
    test_idx = y_test.index
    df_plot = pd.DataFrame({'ds': daily.loc[test_idx, 'ds'], 'actual': y_test.values, 'pred': preds_test})
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_plot['ds'], y=df_plot['actual'], mode='lines+markers', name='Actual'))
    fig.add_trace(go.Scatter(x=df_plot['ds'], y=df_plot['pred'], mode='lines+markers', name='Predicted'))
    fig.update_layout(title=f'Forecast for {product_id} (last fold) - MAE {final_mae:.2f}', xaxis_title='Date', yaxis_title='Quantity')

    return final_model, df_plot, {'cv_mae_mean': float(np.mean(maes)), 'final_mae_on_last_fold': float(final_mae)}

# ------------------ INVENTORY METRICS ------------------
def compute_inventory_metrics_for_product(product_id, y_test, products_df, lead_time_days=14, service_level=SERVICE_LEVEL):
    z = get_z_from_service_level(service_level)
    avg_daily_demand = y_test.mean()
    std_daily_demand = y_test.std()

    safety_stock = z * std_daily_demand * math.sqrt(lead_time_days)
    rop = avg_daily_demand * lead_time_days + safety_stock

    unit_price = products_df.loc[products_df['product_id'] == product_id, 'unit_price'].iloc[0]
    holding_cost = unit_price * HOLDING_COST_RATE
    annual_demand = avg_daily_demand * 365
    if holding_cost <= 0:
        eoq = np.nan
    else:
        eoq = math.sqrt((2 * annual_demand * ORDERING_COST) / holding_cost)

    return {
        'product_id': product_id,
        'avg_daily_demand': float(avg_daily_demand),
        'std_daily_demand': float(std_daily_demand),
        'lead_time_days': int(lead_time_days),
        'safety_stock': float(safety_stock),
        'ROP': float(rop),
        'EOQ': float(eoq)
    }

# ------------------ BUSINESS IMPACT ------------------
def perform_business_impact_analysis(waste_df, stockout_df, products_df):
    results = {}
    if not waste_df.empty:
        wasted_cost = waste_df.merge(products_df[['product_id','unit_price']], on='product_id', how='left')
        wasted_cost['total_cost'] = wasted_cost['wasted_quantity'] * wasted_cost['unit_price']
        total_wasted_cost = wasted_cost['total_cost'].sum()
        results['total_wasted_cost'] = float(total_wasted_cost)
    else:
        results['total_wasted_cost'] = 0.0

    if not stockout_df.empty:
        stockout_cost = stockout_df.merge(products_df[['product_id','unit_price']], on='product_id', how='left')
        stockout_cost['lost_revenue'] = stockout_cost['stockout_quantity'] * stockout_cost['unit_price']
        total_lost_revenue = stockout_cost['lost_revenue'].sum()
        results['total_lost_revenue'] = float(total_lost_revenue)
    else:
        results['total_lost_revenue'] = 0.0

    results['potential_savings'] = results['total_wasted_cost'] * 0.1 + results['total_lost_revenue'] * 0.2
    pd.Series(results).to_csv(os.path.join(OUT_DIR, 'business_impact_summary.csv'))
    return results

# ------------------ APP UI ------------------
def sidebar_controls():
    st.sidebar.header("Simulation parameters")
    num_products = st.sidebar.number_input("Number of products", min_value=50, max_value=2000, value=DEFAULTS['NUM_PRODUCTS'], step=50)
    num_warehouses = st.sidebar.slider("Number of warehouses", 1, 10, DEFAULTS['NUM_WAREHOUSES'])
    num_suppliers = st.sidebar.number_input("Number of suppliers", min_value=5, max_value=200, value=DEFAULTS['NUM_SUPPLIERS'], step=5)
    num_customers = st.sidebar.number_input("Number of customers", min_value=50, max_value=5000, value=DEFAULTS['NUM_CUSTOMERS'], step=50)
    num_purchase_orders = st.sidebar.number_input("Number of purchase orders", min_value=100, max_value=20000, value=DEFAULTS['NUM_PURCHASE_ORDERS'], step=100)
    num_sales_records = st.sidebar.number_input("Number of sales records", min_value=500, max_value=50000, value=DEFAULTS['NUM_SALES_RECORDS'], step=500)
    num_returns_ratio = st.sidebar.slider("Returns ratio (fraction of sales)", 0.0, 0.2, float(DEFAULTS['NUM_RETURNS_RATIO']), 0.01)
    sim_years = st.sidebar.slider("Simulation length (years)", 0.25, 3.0, float(DEFAULTS['SIM_YEARS']), 0.25)
    run_button = st.sidebar.button("Run simulation & analysis")
    return dict(
        num_products=num_products,
        num_warehouses=num_warehouses,
        num_suppliers=num_suppliers,
        num_customers=num_customers,
        num_purchase_orders=num_purchase_orders,
        num_sales_records=num_sales_records,
        num_returns_ratio=num_returns_ratio,
        sim_years=sim_years,
        run_button=run_button
    )

def save_dfs_to_outdir(dfs: dict):
    for name, df in dfs.items():
        if df is None or df.empty:
            continue
        df.to_csv(os.path.join(OUT_DIR, f'{name}.csv'), index=False)

def main():
    st.set_page_config(layout="wide", page_title="Pharma Supply-Chain Simulator")
    st.title("Pharmaceutical Supply-Chain Simulator — Simulation + EDA + Forecast + Optimization")
    st.markdown("Built-in synthetic data generator, day-by-day FEFO inventory simulation, EDA, LightGBM demand forecasting, and inventory metrics (Safety Stock / ROP / EOQ).")

    params = sidebar_controls()

    if not params['run_button']:
        st.info("Adjust parameters in the sidebar then click **Run simulation & analysis**.")
        st.markdown("You can use defaults or increase/decrease sizes for speed. Generated files will be saved to `/mnt/data/outputs`.")
        st.stop()

    # Run simulation pipeline
    with st.spinner("Generating synthetic data..."):
        END_DATE = datetime.now()
        START_DATE = END_DATE - timedelta(days=int(365 * params['sim_years']))

        products_df = generate_products_df(int(params['num_products']))
        warehouses_df = generate_warehouses_df(int(params['num_warehouses']))
        suppliers_df = generate_suppliers_df(int(params['num_suppliers']))

        purchase_orders_df = generate_purchase_orders_df(int(params['num_purchase_orders']), products_df, suppliers_df, START_DATE, END_DATE)
        customers_df = generate_customers_df(int(params['num_customers']))
        sales_df = generate_sales_df(int(params['num_sales_records']), products_df, customers_df, warehouses_df, START_DATE, END_DATE)
        returns_df = generate_returns_df(sales_df, int(params['num_sales_records'] * params['num_returns_ratio']))

        # Save base tables
        base_tables = {
            'products': products_df, 'warehouses': warehouses_df, 'suppliers': suppliers_df,
            'purchase_orders': purchase_orders_df, 'customers': customers_df, 'sales': sales_df, 'returns': returns_df
        }
        save_dfs_to_outdir(base_tables)

    st.success("Data generation complete. Running inventory simulation...")

    with st.spinner("Simulating inventory (FEFO) and computing waste/stockouts..."):
        inventory_snapshot_df, waste_df, stockout_df = simulate_inventory_and_waste(purchase_orders_df, sales_df, returns_df, products_df, warehouses_df)
        if not inventory_snapshot_df.empty:
            inventory_snapshot_df = inventory_snapshot_df.sort_values('on_hand_qty', ascending=False)
        save_dfs_to_outdir({'inventory_snapshot': inventory_snapshot_df, 'waste': waste_df, 'stockouts': stockout_df})

    st.success("Simulation complete. Running EDA and Forecasts...")

    # Layout: three columns for high-level KPIs
    k1, k2, k3 = st.columns(3)
    total_sold = sales_df['sold_quantity'].sum()
    total_waste = waste_df['wasted_quantity'].sum() if not waste_df.empty else 0
    total_stockouts = stockout_df['stockout_quantity'].sum() if not stockout_df.empty else 0
    k1.metric("Total Units Sold", int(total_sold))
    k2.metric("Total Units Wasted (Expired)", int(total_waste))
    k3.metric("Total Units Stockout", int(total_stockouts))

    # EDA plot
    eda_fig = perform_eda(products_df, sales_df, waste_df, stockout_df)
    st.plotly_chart(eda_fig, use_container_width=True)

    # Top products table and chart
    top_products = sales_df.groupby('product_id')['sold_quantity'].sum().nlargest(15).reset_index()
    top_products = top_products.merge(products_df[['product_id','product_name']], on='product_id', how='left')
    top_products['product_name'] = top_products['product_name'].str.slice(0,40)
    st.subheader("Top 15 Bestselling Products")
    st.dataframe(top_products.rename(columns={'sold_quantity':'total_sold'}))
    fig_top = px.bar(top_products, x='product_name', y='sold_quantity', title='Top 15 Products by Sold Quantity')
    st.plotly_chart(fig_top, use_container_width=True)

    # Waste breakdown
    if not waste_df.empty:
        waste_by_reason = waste_df.groupby('reason')['wasted_quantity'].sum().reset_index().sort_values('wasted_quantity', ascending=False)
        fig_waste = px.pie(waste_by_reason, names='reason', values='wasted_quantity', title='Wasted Quantity by Reason')
        st.plotly_chart(fig_waste, use_container_width=True)

    # Stockouts top products
    if not stockout_df.empty:
        stockout_by_product = stockout_df.groupby('product_id')['stockout_quantity'].sum().nlargest(15).reset_index()
        stockout_by_product = stockout_by_product.merge(products_df[['product_id','product_name']], on='product_id', how='left')
        fig_stock = px.bar(stockout_by_product, x='product_name', y='stockout_quantity', title='Top 15 Products with Stockouts')
        st.plotly_chart(fig_stock, use_container_width=True)

    # Forecasting panel
    st.subheader("Demand Forecast (LightGBM)")
    prod_list = sales_df['product_id'].unique().tolist()
    selected_prod = st.selectbox("Select product for forecasting", options=prod_list, index=0)
    if st.button("Run forecast for selected product"):
        with st.spinner("Training forecast model..."):
            try:
                model, df_plot, metrics = train_forecast_for_product(sales_df, selected_prod, n_splits=3)
                st.plotly_chart(px.line(df_plot, x='ds', y=['actual','pred'], labels={'value':'Quantity','variable':'Series'}).update_layout(title=f'Actual vs Predicted for {selected_prod} (last fold)'), use_container_width=True)
                st.write("Forecast metrics:", metrics)

                # inventory metrics based on the y_test series
                inv_metrics = compute_inventory_metrics_for_product(selected_prod, df_plot['actual'], products_df, lead_time_days=14)
                st.write("Inventory metrics (based on last fold):")
                st.json(inv_metrics)
            except Exception as e:
                st.error(f"Forecast failed: {e}")

    # Inventory optimization table for top N products
    st.subheader("Inventory Metrics (Top Products)")
    topN = st.slider("Number of top products to compute metrics for", 3, 30, 10)
    topN_products = sales_df.groupby('product_id')['sold_quantity'].sum().nlargest(topN).index.tolist()
    inv_metrics_list = []
    for pid in topN_products:
        try:
            _, df_plot, _ = train_forecast_for_product(sales_df, pid, n_splits=3)
            metrics = compute_inventory_metrics_for_product(pid, df_plot['actual'], products_df, lead_time_days=14)
            inv_metrics_list.append(metrics)
        except Exception:
            continue
    inv_metrics_df = pd.DataFrame(inv_metrics_list)
    if not inv_metrics_df.empty:
        st.dataframe(inv_metrics_df.sort_values('avg_daily_demand', ascending=False))
        inv_metrics_df.to_csv(os.path.join(OUT_DIR, 'inventory_metrics_top_products.csv'), index=False)

    # Business impact
    st.subheader("Business Impact Summary")
    impact = perform_business_impact_analysis(waste_df, stockout_df, products_df)
    st.write(impact)

    # Download buttons
    st.subheader("Download generated data")
    with st.expander("Available CSV files"):
        files = [f for f in os.listdir(OUT_DIR) if f.endswith('.csv')]
        for f in files:
            file_path = os.path.join(OUT_DIR, f)
            st.download_button(label=f"Download {f}", data=open(file_path, 'rb'), file_name=f)

    st.success(f"All outputs saved to {OUT_DIR}. You can find CSVs and feature importance files there.")

if __name__ == '__main__':
    main()
