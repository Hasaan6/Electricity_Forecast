import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np

# --- Page Config ---
st.set_page_config(page_title="Electricity Forecast", layout="wide")
st.title("🔌 Electricity Consumption Forecasting")

# --- Load Data ---
df = pd.read_csv("data.csv", parse_dates=['date'])
df.set_index('date', inplace=True)
df = df.asfreq('D')
df['consumption'] = df['consumption'].interpolate()
df['temperature'] = df['temperature'].interpolate()

st.subheader("📊 Raw Data")
st.dataframe(df)

# --- Plot Historical ---
st.subheader("📈 Electricity Consumption Over Time")
st.line_chart(df['consumption'])

# --- Forecasting ---
st.subheader("🔮 Forecasted Consumption")

try:
    # --- Future temperature estimate (use last 7 days' avg as placeholder) ---
    avg_temp = df['temperature'][-7:].mean()
    future_temps = pd.Series([avg_temp] * 7, name='temperature')

    # --- Fit SARIMAX with exogenous variable ---
    model = SARIMAX(df['consumption'], exog=df[['temperature']], order=(1,1,1), seasonal_order=(1,1,1,7))
    model_fit = model.fit(disp=False)

    # --- Forecast ---
    forecast_steps = 7
    future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=forecast_steps)
    forecast = model_fit.forecast(steps=forecast_steps, exog=future_temps)

    # --- Combine forecast with actual ---
    forecast_df = pd.DataFrame({'Forecast': forecast}, index=future_dates)
    combined = pd.concat([df['consumption'], forecast_df['Forecast']])

    # --- Plot ---
    st.line_chart(combined)

    # --- Forecast Table ---
    st.write("### 📅 Forecasted Values")
    st.dataframe(forecast_df)

    # --- Pie Chart: Historical vs Forecasted ---
    st.subheader("📊 Historical vs Forecasted Total")
    total_hist = df['consumption'].sum()
    total_fore = forecast_df['Forecast'].sum()

    fig, ax = plt.subplots()
    ax.pie([total_hist, total_fore],
           labels=['Historical', 'Forecasted'],
           autopct='%1.1f%%',
           startangle=90,
           colors=['skyblue', 'orange'])
    ax.axis('equal')
    st.pyplot(fig)

except Exception as e:
    st.error(f"⚠️ Forecasting Error: {e}")
