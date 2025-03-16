import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import pandas as pd

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title("Stock Prediction App")

stocks = ("AAPL", "GOOG", "MSFT", "GME", "AMZN", "TSLA")
selected_stock = st.selectbox("Select dataset for prediction", stocks)

n_years = st.slider("Years of prediction:", 1, 4)
period = n_years * 365

@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text('Load data...')
data = load_data(selected_stock)
data_load_state.text('Loading data...done!')
st.subheader('Raw data')
st.write(data.tail())

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
    fig.layout.update(title_text='Time Series data', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

# Prepare data for forecasting
df_train = pd.DataFrame()
df_train['ds'] = data['Date']
df_train['y'] = data['Close'].astype(float)

# Create and train the Prophet model
m = Prophet(daily_seasonality=True)
m.fit(df_train)

# Make future predictions
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# Show forecast data
st.subheader('Forecast data')
st.write(forecast.tail())

# Plot forecast
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

# Show forecast components
fig2 = m.plot_components(forecast)
st.write(fig2)

