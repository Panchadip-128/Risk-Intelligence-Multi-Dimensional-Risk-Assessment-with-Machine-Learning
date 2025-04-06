from matplotlib import pyplot as plt
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns

def get_clean_data():
    data = pd.read_csv("diabetes.csv")
    return data

def diabetes_sidebar():
    st.sidebar.header("Diabetes Measurements")
    
    data = get_clean_data()
    
    slider_labels = [
        ("Pregnancies", "Pregnancies"),
        ("Glucose", "Glucose"),
        ("Blood Pressure", "BloodPressure"),
        ("Skin Thickness", "SkinThickness"),
        ("Insulin", "Insulin"),
        ("BMI", "BMI"),
        ("Diabetes Pedigree Function", "DiabetesPedigreeFunction"),
        ("Age", "Age")
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
    
    X = data.drop(['Outcome'], axis=1)
    
    scaled_dict = {}
    
    for key, value in input_dict.items():
        max_val = float(X[key].max())
        min_val = float(X[key].min())
        scaled_value = (value - min_val) / (max_val - min_val)
        scaled_dict[key] = scaled_value
    
    return scaled_dict

def get_radar_chart_diabetes(input_data):
    input_data = get_scaled_values(input_data)
    
    categories = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

    values = [input_data[category] for category in categories]

    fig = px.bar(
        x=categories,
        y=values,
        labels={'x': 'Features', 'y': 'Scaled Values'},
        title="Diabetes Measurements"
    )
    
    return fig
def get_parallel_coordinates_plot(input_data):
    input_data = get_scaled_values(input_data)
    
    categories = list(input_data.keys())
    values = list(input_data.values())
    
    # Create a DataFrame for Plotly express
    df = pd.DataFrame([values], columns=categories)
    
    fig = px.parallel_coordinates(
        df,
        dimensions=categories,
        title="Diabetes Measurements - Parallel Coordinates Plot"
    )
    
    return fig
def display_correlation_matrix(data):
       data = get_clean_data()
       corr = data.corr()
    
       fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.index.values,
        y=corr.columns.values,
        colorscale='Viridis'))
    
       fig.update_layout(
        title="Correlation Matrix",
        autosize=False,
        width=600,
        height=600,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=40, r=40, b=85, t=100),
    )
    
       return fig
    
def get_line_chart_diabetes(input_data):
    data = get_clean_data()
    data.drop(columns=['Outcome'], inplace=True)

    
    # Calculate mean and standard deviation values
    mean_values = data.mean()
    std_values = data.std()

    # Scale the mean and standard deviation values
    scaled_mean_values = get_scaled_values(mean_values.to_dict())
    scaled_std_values = get_scaled_values(std_values.to_dict())
    scaled_real_time_values = get_scaled_values(input_data)

    categories = data.columns.tolist()

    # Prepare data for plotting
    x_values = categories * 3
    y_values = list(scaled_mean_values.values()) + list(scaled_std_values.values()) + list(scaled_real_time_values.values())
    line_type = ['Mean'] * len(categories) + ['Standard Deviation'] * len(categories) + ['Real-Time'] * len(categories)

    # Plotting with Plotly Express
    fig = px.line(
        x=x_values,
        y=y_values,
        color=line_type,
        labels={'x': 'Features', 'y': 'Scaled Values'},
        title="Diabetes Measurements - Mean, Standard Deviation (SD), and Real-Time Values"
    )

    fig.update_layout(
        xaxis={'categoryorder': 'array', 'categoryarray': categories},  # Ensure correct category order
        legend_title='Values',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )

    return fig

def diabetes_predictions(input_data):
    model = pickle.load(open("model/diabetes_model.pkl", "rb"))
    scaler = pickle.load(open("model/diabetes_scaler.pkl", "rb"))
    
    input_array = np.array(list(input_data.values())).reshape(1, -1)
    
    input_array_scaled = scaler.transform(input_array)
    
    prediction = model.predict(input_array_scaled)
    
    st.subheader("Diabetes Prediction")
    st.write("The diabetes status is:")
    predicted_prob_disease = model.predict_proba(input_array_scaled)[0][1]
    if predicted_prob_disease >= 0.5:
        st.write("<span class='diagnosis malicious'>Disease Present</span>", unsafe_allow_html=True)
    else:
        st.write("<span class='diagnosis benign'>No Disease</span>", unsafe_allow_html=True)

    
    st.write("Probability of no disease: ", model.predict_proba(input_array_scaled)[0][0])
    st.write("Probability of disease: ", model.predict_proba(input_array_scaled)[0][1])
    
    st.warning("This app can assist medical professionals in making a diagnosis, but should not be used as a substitute for a professional diagnosis.")


def main():
    st.set_page_config(
        page_title="Diabetes Predictor",
        page_icon=":syringe:",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    with open("assets/style.css") as f:
        st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)
    
    input_data = diabetes_sidebar()
    
    with st.container():
        st.title("Diabetes Predictor")
        st.write("This app predicts whether a patient has diabetes based on various measurements. You can update the measurements using the sliders in the sidebar.")
    
    col1, col2 = st.columns([4,1])
    
    with col1:
        radar_chart = get_radar_chart_diabetes(input_data)
        line_chart = get_line_chart_diabetes(input_data)
        st.plotly_chart(radar_chart)
        st.plotly_chart(line_chart)
    with col2:
        diabetes_predictions(input_data)

if __name__ == '__main__':
    main()
