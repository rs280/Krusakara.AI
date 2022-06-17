import numpy as np
import pandas as pd
import streamlit as st
import neuralprophet as NeuralProphet
import plotly.graph_objects as go

st.set_page_config(
   page_title="କୃଷକ.AI  ",
   page_icon="favicon.ico")
st.title('Welcome to କୃଷକ.AI')

st.subheader('First, go to the "Crop Prediction" page to see the prediction of the crop based on your soil.')

st.subheader('Second, go to the "Future Predictor" page to see the prediction of the future price of the crop you got.')



data_location = "https://raw.githubusercontent.com/ourownstory/neuralprophet-data/main/datasets/"

df = pd.read_csv(data_location + 'wp_log_peyton_manning.csv')

m = NeuralProphet()
metrics = m.fit(df)
future = m.make_future_dataframe(df=df, periods=365)
forecast = m.predict(df=future)
fig_forecast = m.plot(forecast)
st.plotly_chart(fig_forecast)