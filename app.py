import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pmdarima import auto_arima
from datetime import timedelta

# Streamlit page settings
st.set_page_config(page_title="⚡ Electricity Forecast App", layout="wide")
sns.set_style("whitegrid")

# Load and preprocess data
@st.cache_data
def load_data():
    df = pd.read_csv("data.csv")
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    df = df.set_index('date')
    df = df.fillna(method='ffill')
    return df

df = load_data()

# Sidebar controls
st.sidebar.title("⚙ Controls")
households = df['household'].unique()
selected_household = st.sidebar.selectbox("Select Household", households)
temp_min, temp_max = st.sidebar.slider("Temperature Range (°C)",
                                       float(df['temperature'].min()),
                                       float(df['temperature'].max()),
                                       (float(df['temperature'].min()), float(df['temperature'].max())))
n_days = st.sidebar.slider("Days to Forecast", 7, 60, 14)

# Filter data
filtered_df = df[(df['household'] == selected_household) &
                 (df['temperature'] >= temp_min) &
                 (df['temperature'] <= temp_max)]

# Title
st.title("⚡ Advanced Electricity Consumption Forecasting")
st.markdown(f"Forecasting for **Household {selected_household}** based on historical trends.")

# Summary statistics
col1, col2, col3 = st.columns(3)
col1.metric("Average Consumption", f"{filtered_df['consumption'].mean():.2f} kWh")
col2.metric("Max Consumption", f"{filtered_df['consumption'].max():.2f} kWh")
col3.metric("Min Consumption", f"{filtered_df['consumption'].min():.2f} kWh")

# Moving average for trend
filtered_df['MA7'] = filtered_df['consumption'].rolling(window=7).mean()

# Historical plot
st.subheader("📊 Historical Consumption & Trend")
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(filtered_df.index, filtered_df['consumption'], label="Consumption", color='blue')
ax.plot(filtered_df.index, filtered_df['MA7'], label="7-Day Moving Avg", color='orange')
ax.set_ylabel("kWh")
ax.legend()
st.pyplot(fig)

# Anomaly detection (high usage > mean + 2*std)
threshold = filtered_df['consumption'].mean() + 2 * filtered_df['consumption'].std()
anomalies = filtered_df[filtered_df['consumption'] > threshold]

# Show anomalies if any
if not anomalies.empty:
    st.warning(f"⚠ High-usage days detected: {len(anomalies)} days")
    st.dataframe(anomalies)

# Model training
model = auto_arima(filtered_df['consumption'], seasonal=True, m=7, suppress_warnings=True)
model.fit(filtered_df['consumption'])

# Forecasting
future_dates = pd.date_range(start=filtered_df.index[-1] + timedelta(days=1), periods=n_days)
forecast = model.predict(n_periods=n_days)
forecast_df = pd.DataFrame({'date': future_dates, 'forecast': forecast})
forecast_df = forecast_df.set_index('date')

# Forecast plot
st.subheader(f"🔮 {n_days}-Day Forecast")
fig2, ax2 = plt.subplots(figsize=(12, 5))
ax2.plot(filtered_df.index, filtered_df['consumption'], label="Historical", color='blue')
ax2.plot(forecast_df.index, forecast_df['forecast'], label="Forecast", color='red', linestyle='--')
ax2.set_ylabel("kWh")
ax2.legend()
st.pyplot(fig2)

# Forecast table
st.subheader("📋 Forecast Table")
st.dataframe(forecast_df.style.format("{:.2f}"))

# Download button
csv = forecast_df.to_csv().encode('utf-8')
st.download_button("⬇ Download Forecast Data", data=csv, file_name="forecast.csv", mime="text/csv")

# Footer
st.markdown("---")
st.caption("Created with ❤️ using Streamlit & Auto-ARIMA")
