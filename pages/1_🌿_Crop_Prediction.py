import numpy as np
import pandas as pd
import streamlit as st
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.figure_factory as ff

model = pickle.load(open('pages/model.pkl', 'rb'))
df = pd.read_csv("pages/crop_recommendation.csv")

converts_dict = {
    'Nitrogen': 'N',
    'Phosphorus': 'P',
    'Potassium': 'K',
    'Temperature': 'temperature',
    'Humidity': 'humidity',
    'Rainfall': 'rainfall',
    'ph': 'ph'
}

def predict_crop(n, p, k, temperature, humidity, ph, rainfall):
    input = np.array([[n, p, k, temperature, humidity, ph, rainfall]]).astype(np.float64)
    prediction = model.predict(input)
    return prediction[0]

def scatterPlotDrawer(x,y):
    fig = plt.figure(figsize=(20,15))
    sns.set_style("whitegrid")
    sns.scatterplot(data=df, x=x, y=y, hue="label", size="label", palette="deep", sizes=(20, 200), legend="full")
    plt.xlabel(x, fontsize=22)
    plt.ylabel(y, fontsize=22)
    plt.xticks(rotation=90, fontsize=18)
    plt.legend(prop={'size': 18})
    plt.yticks(fontsize=16)
    st.write(fig)

def barPlotDrawer(x,y):
    fig = plt.figure(figsize=(20,15))
    sns.set_style("whitegrid")
    sns.barplot(data=df, x=x, y=y)
    plt.xlabel("Crops", fontsize=22)
    plt.ylabel(y, fontsize=22)
    plt.xticks(rotation=90, fontsize=18)
    plt.legend(prop={'size': 18})
    plt.yticks(fontsize=16)
    st.write(fig)

def boxPlotDrawer(x,y):
    fig = plt.figure(figsize=(20,15))
    sns.set_style("whitegrid")
    sns.boxplot(x=x, y=y, data=df)
    sns.despine(offset=10, trim=True)
    plt.xlabel("Crops", fontsize=22)
    plt.ylabel(y, fontsize=22)
    plt.xticks(rotation=90, fontsize=18)
    plt.legend(prop={'size': 18})
    plt.yticks(fontsize=16)
    st.write(fig)

def main():
    html_temp_vis = """
    <div style="background-color:#025246 ;padding:10px;margin-bottom:30px">
    <h2 style="color:white;text-align:center;"> Visualize Soil Properties </h2>
    </div>
    """

    html_temp_pred = """
    <div style="background-color:#025246 ;padding:10px;margin-bottom:30px">
    <h2 style="color:white;text-align:center;"> Which Crop To Cultivate? </h2>
    </div>
    """

   
    
  
    st.markdown(html_temp_pred, unsafe_allow_html=True)
    st.header("To predict your crop give values")
    st.subheader("Drag to Give Values")
    n = st.slider('Nitrogen', 0, 140)
    p = st.slider('Phosphorus', 0, 145)
    k = st.slider('Potassium', 0, 205)
    temperature = st.slider('Temperature in Celsius', 0, 100)
    humidity = st.slider('Humidity', 0, 100)
    ph = st.slider('pH', 0, 14)
    rainfall = st.slider('Rainfall', 0, 400)
        
    if st.button("Predict your crop"):
            output=predict_crop(n, p, k, temperature, humidity, ph, rainfall)
            res = "“"+ output.capitalize() + "”"
            st.success('The most suitable crop for your field is {}'.format(res))

if __name__=='__main__':
    main()