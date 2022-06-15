import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
   page_title="AgriAI App",
   page_icon="favicon.ico")
st.title('Welcome to AgriAI App')

st.subheader('First, go to the "Crop Prediction" page to see the prediction of the crop based on your soil.')

st.subheader('Second, go to the "Future Predictor" page to see the prediction of the future price of the crop you got.')