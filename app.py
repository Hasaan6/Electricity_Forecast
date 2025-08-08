import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX

st.title("🔌 Electricity Consumption Forecast")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Convert to datetime and sort
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    df.set_index('date', inplace=True)

    st.subheader("📊 Original Data")
    st.line_chart(df['consumption'])

    # Train-test split
    train = df.iloc[:-7]
    test = df.iloc[-7:]

    # Model training
    model = SARIMAX(train['consumption'], order=(1,1,1), seasonal_order=(1,1,1,7))
    result = model.fit(disp=False)

    # Forecast
    forecast = result.forecast(steps=7)
    forecast.index = test.index

    # Plot
    st.subheader("🔮 Forecast vs Actual")
    fig, ax = plt.subplots()
    train['consumption'].plot(ax=ax, label="Train")
    test['consumption'].plot(ax=ax, label="Actual")
    forecast.plot(ax=ax, label="Forecast")
    ax.legend()
    st.pyplot(fig)

    st.success("Forecast complete ✅")
