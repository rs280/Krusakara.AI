


import streamlit as st
import pandas_datareader as pdr
import pandas as pd
from neuralprophet import NeuralProphet
import plotly.graph_objs as go


st.title('Futures Price History & Prediction App:')

stocks = ['Wheat','Rice','Corn', 'Oat', 'Soybean','Cocoa','Coffee','Cotton','Sugar']
selected_stocks = st.selectbox("Select Your Future", stocks)

START = st.date_input('Start', value=pd.to_datetime("2017-01-01"))
TODAY = st.date_input('End(Today)', value=pd.to_datetime("today"))

n_years = st.slider('Days of prediction:', 1, 365, 365)
period = n_years * 1


@st.cache
def load_data(ticker):
    data = pdr.get_data_yahoo(ticker, start=START, end=TODAY)
    data.reset_index(inplace=True)
    return data


data_load_state = st.text("Loading Data...")
if selected_stocks == 'Wheat':
    data = load_data('ZW=F')
if selected_stocks == 'Rice':
    data = load_data('ZR=F')
if selected_stocks == 'Corn':
    data = load_data('ZC=F')
if selected_stocks == 'Oat':
    data = load_data('ZO=F')
if selected_stocks == 'Soybean':
    data = load_data('ZS=F')
if selected_stocks == 'Cocoa':
    data = load_data('CC=F')
if selected_stocks == 'Coffee':
    data = load_data('KC=F')
if selected_stocks == 'Cotton':
    data = load_data('CT=F')
if selected_stocks == 'Sugar':
    data = load_data('SB=F')
data_load_state.text('Loading Data...Done!')

st.subheader('Raw Data (Last 7 Days):')
st.write(data.tail(7))


def plot_raw_date():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='Actual Close Price'))
    fig.layout.update(title_text='Time Series Data', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)


plot_raw_date()

# Forecasting

df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={'Date': 'ds', 'Close': 'y'})
size = len(df_train)
model = NeuralProphet()
## split teh data into train and test


model.fit(df_train)

future = model.make_future_dataframe(df=df_train,periods=period)
forecast = model.predict(df=future)

st.subheader('Raw Predicted Data')
st.write(forecast.tail(7))


# # Plotting predicted data


fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat1'], name='Predicted Price'))
fig2.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='Adj Close Price'))
fig2.layout.update(title_text='Predicted Price', xaxis_rangeslider_visible=True)
st.plotly_chart(fig2) 

if (forecast.at[forecast.index[-1], 'yhat1']>data['Close'][size]):
    st.write('The predicted price will be higher than the current price')
    price = forecast['yhat1'].iloc[-1]-data['Close'][size]
    st.write('The predicted price will be higher than the current price by', price)
    land = st.slider('How many acres of land do you own?', 0, 100, 50)
    st.write('If you use all of your land to harvest wheat, you will earn', (land*40/5000)*forecast['yhat'][n_years+1370] , 'in total.')
elif (forecast['yhat1'].iloc[-1]<data['Close'][size]):
    st.write('The predicted price will be lower than the current price')
    price = data['Close'][size]-forecast['yhat1'].iloc[-1]
    st.write('The predicted price will be lower than the current price by $', price)
    st.subheader('How many acres of  land do you own?')
    land = st.slider('', 0, 100, 50)
    st.write('It is not advisable to harvest wheat')
    
else:
    st.write('The predicted price will be the same as the current price')


