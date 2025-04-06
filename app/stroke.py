import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

def get_clean_data():
    data = pd.read_csv("healthcare-dataset-stroke-data.csv")
    data.drop(data[data['gender'] == 'Other'].index, inplace=True)
    return data

def stroke_disease_sidebar():
    st.sidebar.header("Stroke Disease Predictor")

    data = get_clean_data()

    slider_labels = [
        ("Gender", "gender"),
        ("Age", "age"),
        ("Hypertension", "hypertension"),
        ("Heart Disease", "heart_disease"),
        ("Ever Married", "ever_married"),
        ("Work Type", "work_type"),
        ("Residence Type", "Residence_type"),
        ("Average Glucose Level", "avg_glucose_level"),
        ("BMI", "bmi"),
        ("Smoking Status", "smoking_status")
    ]

    input_dict = {}

    for label, key in slider_labels:
        if key == "gender":
            input_dict[key] = st.sidebar.selectbox(label, options=["Male", "Female"])
        elif key == "ever_married":
            input_dict[key] = st.sidebar.selectbox(label, options=["Yes", "No"])
        elif key == "work_type":
            input_dict[key] = st.sidebar.selectbox(label, options=["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
        elif key == "Residence_type":
            input_dict[key] = st.sidebar.selectbox(label, options=["Urban", "Rural"])
        elif key == "smoking_status":
            input_dict[key] = st.sidebar.selectbox(label, options=["never smoked", "formerly smoked", "smokes", "Unknown"])
        else:
            input_dict[key] = st.sidebar.slider(
                label,
                min_value=float(data[key].min()),
                max_value=float(data[key].max()),
                value=float(data[key].mean())
            )

    return input_dict

def preprocess_input(input_dict):
    df = get_clean_data()
    df = df.drop(columns=["stroke"])

    input_df = pd.DataFrame([input_dict])
    df = pd.concat([df, input_df], ignore_index=True)
    
    object_cols = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
    label_encoder = LabelEncoder()
    for col in object_cols:
        df[col] = label_encoder.fit_transform(df[col])
    
    imputer = SimpleImputer(strategy='most_frequent')
    df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    
    scaler = pickle.load(open("model/stroke_scaler.pkl", "rb"))
    df_scaled = scaler.transform(df)
    
    return df_scaled[-1].reshape(1, -1)

def stroke_disease_predictions(input_data):
    model = pickle.load(open("model/stroke_model.pkl", "rb"))
    input_array=preprocess_input(input_data)
    prediction = model.predict(input_array)
    st.subheader("Stroke Disease Prediction")
    st.write("The stroke disease status is:")
    
    prediction = model.predict(input_array)
    probabilities = model.predict_proba(input_array)[0]
    probability_disease_present = probabilities[1]
    probability_no_disease = probabilities[0]
    probability_disease = probabilities[1]

    if probability_disease_present >= 0.5:
        st.write("<span class='diagnosis malicious'>Disease Present</span>", unsafe_allow_html=True)
    else:
        st.write("<span class='diagnosis benign'>No Disease</span>", unsafe_allow_html=True)
    st.write("Probability of no disease: ", probability_no_disease)
    st.write("Probability of disease:               ", probability_disease)
    
    st.warning("This app can assist medical professionals in making a diagnosis, but should not be used as a substitute for a professional diagnosis.")
def get_bar_chart(input_data): # Debugging statement
    categories = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi']
    values = input_data

    fig = px.bar(
        x=categories,
        y=values,
        labels={'x': 'Features', 'y': 'Values'},
        title="Stroke Disease Measurements"
    )
    
    return fig

def get_line_chart_stroke(input_data):
    preprocessed_data = preprocess_input(input_data)

    st.write("Debug: input_data shape:", preprocessed_data.shape)  # Debugging statement
    categories = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi']
    values = preprocessed_data.flatten()[:len(categories)]

    fig = px.line(
        x=categories,
        y=values,
        labels={'x': 'Features', 'y': 'Values'},
        title="Stroke Disease Measurements - Line Chart"
    )
    
    return fig
def pie_chart():
    df=get_clean_data()
    job = df.groupby(df['work_type'])['stroke'].sum()
    df_job = pd.DataFrame({'labels': job.index,
                   'values': job.values
                  })
    colors2= ['palegreen','paleturquoise','thistle','moccasin']
    df_job.iplot(kind='pie',labels='labels',values='values', title='Work type of people who had stroke', colors = colors2, 
                pull=[0.1, 0.1, 0.1, 0.2])
def generate_pie_charts(data):
    # Filter data for people who had a stroke
    stroke_data = data[data['stroke'] == 1]

    # Work type of people who had a stroke
    work_type_fig = px.pie(
        stroke_data,
        names='work_type',
        title='Work Type of People Who Had Stroke',
        hole=0.4,
      
    )

    # The proportion of stroke among gender
    gender_fig = px.pie(
        stroke_data,
        names='gender',
        title='Proportion of Stroke Among Gender',
        hole=0.4
    )

    # Smoking status of people who had a stroke
    smoking_status_fig = px.pie(
        stroke_data,
        names='smoking_status',
        title='Smoking Status of People Who Had Stroke',
        hole=0.4
    )

    # Residence area of people who had a stroke
    residence_type_fig = px.pie(
        stroke_data,
        names='Residence_type',
        title='Residence Area of People Who Had Stroke',
        hole=0.4
    )

    return work_type_fig, gender_fig, smoking_status_fig, residence_type_fig
def main():
    st.set_page_config(
        page_title="Stroke Disease Predictor",
        page_icon=":brain:",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    with open("assets/style.css") as f:
        st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)
    
    input_data = stroke_disease_sidebar()
    preprocessed_data = preprocess_input(input_data)
    
    st.write("Debug: preprocessed_data shape in main:", preprocessed_data.shape)  # Debugging statement
    
    with st.container():
        st.title("Stroke Disease Predictor")
        st.write("This app predicts whether a patient has stroke disease based on various measurements. You can update the measurements using the sliders in the sidebar.")
    
    col1, col2 = st.columns([4,1])
    
    with col1:
        bar_chart = get_bar_chart(preprocessed_data)
        st.plotly_chart(bar_chart)
    with col2:
        stroke_disease_predictions(preprocessed_data)

if __name__ == '__main__':
    main()