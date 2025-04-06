from matplotlib import pyplot as plt
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns


def get_clean_data():
    df=pd.read_csv('heart.csv')
    df.drop_duplicates(inplace=True)
    df.fillna(df.mean(), inplace=True)
    return df

def add_sidebar_heart():
    st.sidebar.header("Heart Disease Features")
  
    data = get_clean_data()
  
    slider_labels = [
        ("Age", "age"),
        ("Sex", "sex"),
        ("Chest Pain Type (cp)", "cp"),
        ("Resting Blood Pressure (trestbps)", "trestbps"),
        ("Serum Cholesterol (chol)", "chol"),
        ("Fasting Blood Sugar > 120 mg/dl (fbs)", "fbs"),
        ("Resting Electrocardiographic Results (restecg)", "restecg"),
        ("Maximum Heart Rate Achieved (thalach)", "thalach"),
        ("Exercise Induced Angina (exang)", "exang"),
        ("ST Depression Induced by Exercise (oldpeak)", "oldpeak"),
        ("Slope of the Peak Exercise ST Segment (slope)", "slope"),
        ("Number of Major Vessels (ca)", "ca"),
        ("Thalassemia (thal)", "thal")
    ]

    input_dict = {}

    for label, key in slider_labels:
        input_dict[key] = st.sidebar.slider(
            label,
            min_value=float(0),
            max_value=float(data[key].max()),
            value=float(data[key].mean())
        )
    
    return input_dict

def get_scaled_values(input_dict):
  data = get_clean_data()
  X = data.drop(['target'], axis=1)
  scaled_dict = {}
  for key, value in input_dict.items():
    max_val = X[key].max()
    min_val = X[key].min()
    scaled_value = (value - min_val) / (max_val - min_val)
    scaled_dict[key] = scaled_value
  
  return scaled_dict

def heart_disease_predictions(input_data):
    # Load the model and scaler
    model = pickle.load(open("model/heart_model.pkl", "rb"))
    scaler = pickle.load(open("model/heart_scaler.pkl", "rb"))
    
    # Prepare the input data for prediction
    input_array = np.array(list(input_data.values())).reshape(1, -1)
    
    # Scale the input data
    input_array_scaled = scaler.transform(input_array)
    
    # Make the prediction
    prediction = model.predict(input_array_scaled)
    
    # Display the prediction results
    st.subheader("Heart Disease Prediction")
    st.write("The heart disease status is:")
    predicted_prob_disease = model.predict_proba(input_array_scaled)[0][1]
    if predicted_prob_disease >= 0.5:
        st.write("<span class='diagnosis malicious'>Disease Present</span>", unsafe_allow_html=True)
    else:
        st.write("<span class='diagnosis benign'>No Disease</span>", unsafe_allow_html=True)

    # Display the probabilities
    st.write("Probability of no disease: ", model.predict_proba(input_array_scaled)[0][0])
    st.write("Probability of disease: ", model.predict_proba(input_array_scaled)[0][1])
    
    # Disclaimer
    st.info("This app can assist medical professionals in making a diagnosis, but should not be used as a substitute for a professional diagnosis.")

def get_scaled_values1(data):
    data=get_clean_data()
    data= data.drop(['target'], axis=1)

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    scaled_df = pd.DataFrame(scaled_data, columns=data.columns)
    return scaled_df

def get_radar_chart_heart(input_data):
    # Assuming input_data is already scaled
    input_data = get_scaled_values(input_data)
    
    # Assuming `get_clean_data` provides the dataset
    data1 = get_clean_data()
    data1=data1.drop('target',axis=1)
    data=get_scaled_values1(data1)
    #data=data.drop('target', axis=1)

    
    # Calculate mean and standard deviation for each category
    mean_values = data.mean()
    std_values = data.std()

    categories = ['Age', 'Sex', 'Chest Pain Type', 'Resting BP', 'Cholesterol', 
                  'Fasting Blood Sugar', 'Resting ECG', 'Max Heart Rate', 
                  'Exercise Induced Angina', 'ST Depression', 'Slope of ST', 
                  'Major Vessels', 'Thalassemia']

    fig = go.Figure()

    # Add trace for real-time input values
    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['age'], input_data['sex'], input_data['cp'], 
            input_data['trestbps'], input_data['chol'], input_data['fbs'], 
            input_data['restecg'], input_data['thalach'], input_data['exang'], 
            input_data['oldpeak'], input_data['slope'], input_data['ca'], 
            input_data['thal']
        ],
        theta=categories,
        fill='toself',
        name='Real-time Values'
    ))

    # Add trace for mean values
    fig.add_trace(go.Scatterpolar(
        r=[
            mean_values['age'], mean_values['sex'], mean_values['cp'], 
            mean_values['trestbps'], mean_values['chol'], mean_values['fbs'], 
            mean_values['restecg'], mean_values['thalach'], mean_values['exang'], 
            mean_values['oldpeak'], mean_values['slope'], mean_values['ca'], 
            mean_values['thal']
        ],
        theta=categories,
        fill='toself',
        name='Mean Values'
    ))

    # Add trace for mean + standard deviation values
    fig.add_trace(go.Scatterpolar(
        r=[
            std_values['age'],  std_values['sex'], std_values['cp'], 
            std_values['trestbps'],  std_values['chol'], std_values['fbs'],  std_values['restecg'], std_values['thalach'],  std_values['exang'], 
            std_values['oldpeak'], std_values['slope'],  std_values['ca'], 
            std_values['thal']
        ],
        theta=categories,
        fill='toself',
        name='Mean + Std Dev'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]  # Adjust range if necessary
            )),
        showlegend=True
    )
    
    return fig

def plot_line_chart_heart(input_data):
    # Assuming `get_scaled_values` scales the input data
    input_data = get_scaled_values(input_data)
    
    # Assuming `get_clean_data` provides the dataset
    data1 = get_clean_data()
    data=get_scaled_values1(data1)
    
    # Calculate mean and standard deviation for each category
    mean_values = data.mean()
    std_values = data.std()

    categories = ['Age', 'Sex', 'Chest Pain Type', 'Resting BP', 'Cholesterol', 
                  'Fasting Blood Sugar', 'Resting ECG', 'Max Heart Rate', 
                  'Exercise Induced Angina', 'ST Depression', 'Slope of ST', 
                  'Major Vessels', 'Thalassemia']

    real_time_values = [
        input_data['age'], input_data['sex'], input_data['cp'], 
        input_data['trestbps'], input_data['chol'], input_data['fbs'], 
        input_data['restecg'], input_data['thalach'], input_data['exang'], 
        input_data['oldpeak'], input_data['slope'], input_data['ca'], 
        input_data['thal']
    ]
    
    mean_values_list = [
        mean_values['age'], mean_values['sex'], mean_values['cp'], 
        mean_values['trestbps'], mean_values['chol'], mean_values['fbs'], 
        mean_values['restecg'], mean_values['thalach'], mean_values['exang'], 
        mean_values['oldpeak'], mean_values['slope'], mean_values['ca'], 
        mean_values['thal']
    ]
    
    mean_plus_std_values = [
        mean_values['age'] + std_values['age'], mean_values['sex'] + std_values['sex'], mean_values['cp'] + std_values['cp'], 
        mean_values['trestbps'] + std_values['trestbps'], mean_values['chol'] + std_values['chol'], mean_values['fbs'] + std_values['fbs'], 
        mean_values['restecg'] + std_values['restecg'], mean_values['thalach'] + std_values['thalach'], mean_values['exang'] + std_values['exang'], 
        mean_values['oldpeak'] + std_values['oldpeak'], mean_values['slope'] + std_values['slope'], mean_values['ca'] + std_values['ca'], 
        mean_values['thal'] + std_values['thal']
    ]
    df = pd.DataFrame({
        'Features': categories * 3,
        'Values': real_time_values + mean_values_list + mean_plus_std_values,
        'Type': ['Real-time Values'] * len(categories) + ['Mean Values'] * len(categories) + ['Mean + Std Dev'] * len(categories)
    })

    # Plotting using plotly.express
    fig = px.line(df, x='Features', y='Values', color='Type', 
                  labels={'Values': 'Scaled Values', 'Features': 'Features'},
                  title="Heart Disease Features - Line Chart")
 
    fig.update_layout(
        xaxis={'categoryorder': 'array', 'categoryarray': categories},  # Ensure correct category order
        legend_title='Values',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )

    return fig




   