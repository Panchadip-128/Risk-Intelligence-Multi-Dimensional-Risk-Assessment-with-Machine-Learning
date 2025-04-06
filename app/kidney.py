import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px

def get_clean_data():
    data = pd.read_csv("kidney_disease.csv")
    # Dropping unnecessary columns
    columns_to_drop = ["id", "age", "Red_Blood_Cells", "Pus_Cell_Clumps", "Serum_Creatinine", "Potassium", "Coronary_Artery_Disease", "Bacteria"]
    existing_columns_to_drop = [col for col in columns_to_drop if col in data.columns]
    data = data.drop(existing_columns_to_drop, axis=1)
    
    # Handling categorical data
    categorical_columns = ["htn", "dm", "cad", "appet", "pe", "ane", "classification"]
    for column in categorical_columns:
        data[column] = data[column].map({"Yes": 1, "No": 0, "ckd": 1, "notckd": 0})
    
    # Ensure numeric columns are converted to float and handle missing values
    for column in data.columns:
        if column not in categorical_columns:
            data[column] = pd.to_numeric(data[column], errors='coerce')
            data[column] = data[column].fillna(data[column].mean())
    
    return data

def kidney_disease_sidebar():
    st.sidebar.header("Kidney Disease Measurements")
    
    data = get_clean_data()
    
    slider_labels = [
        ("Blood Pressure", "bp"),
        ("Specific Gravity", "sg"),
        ("Albumin", "al"),
        ("Sugar", "su"),
        ("Blood Glucose Random", "bgr"),
        ("Blood Urea", "bu"),
        ("Sodium", "sod"),
        ("Hemoglobin", "hemo"),
        ("Hypertension", "htn"),
        ("Diabetes Mellitus", "dm"),
        ("Appetite", "appet"),
        ("Pedal Edema", "pe"),
        ("Anemia", "ane"),
        ("Classification", "classification")
    ]

    input_dict = {}

    for label, key in slider_labels:
        if key in ["htn", "dm", "appet", "pe", "ane"]:
            input_dict[key] = st.sidebar.selectbox(label, options=[1, 0], format_func=lambda x: 'Yes' if x == 1 else 'No')
        elif key == "classification":
            input_dict[key] = st.sidebar.selectbox(label, options=[1, 0], format_func=lambda x: 'ckd' if x == 1 else 'notckd')
        else:
            input_dict[key] = st.sidebar.slider(
                label,
                min_value=float(data[key].min()),
                max_value=float(data[key].max()),
                value=float(data[key].mean())
            )
    
    return input_dict

def get_scaled_values(input_dict):
    data = get_clean_data()
    
    X = data.drop(['classification'], axis=1)
    
    scaled_dict = {}
    
    for key, value in input_dict.items():
        if key in ["htn", "dm", "appet", "pe", "ane", "classification"]:
            scaled_dict[key] = 1 if value == "Yes" else 0
        else:
            max_val = float(X[key].max())
            min_val = float(X[key].min())
            scaled_value = (value - min_val) / (max_val - min_val)
            scaled_dict[key] = scaled_value
    
    return scaled_dict

def get_correlation_matrix(input_data):
    input_data = get_scaled_values(input_data)
    
    df = pd.DataFrame([input_data])
    fig = px.imshow(df.corr(), text_auto=True, title="Correlation Matrix")
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    
    return fig

def kidney_disease_predictions(input_data):
    model = pickle.load(open("model/kidney_model.pkl", "rb"))
    scaler = pickle.load(open("model/kidney_scaler.pkl", "rb"))
    
    input_array = np.array(list(input_data.values())).reshape(1, -1)
    
    input_array_scaled = scaler.transform(input_array)
    
    prediction = model.predict(input_array_scaled)
    
    st.subheader("Kidney Disease Prediction")
    st.write("The kidney disease status is:")
    predicted_prob_disease = model.predict(input_array_scaled)[0]
    if predicted_prob_disease == "ckd":
        st.write("<span class='diagnosis malicious'>Disease Present</span>", unsafe_allow_html=True)
    else:
        st.write("<span class='diagnosis benign'>No Disease</span>", unsafe_allow_html=True)

    st.write("This app can assist medical professionals in making a diagnosis, but should not be used as a substitute for a professional diagnosis.")

def main():
    st.set_page_config(
        page_title="Kidney Disease Predictor",
        page_icon=":kidney:",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    with open("assets/style.css") as f:
        st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)
    
    input_data = kidney_disease_sidebar()
    
    with st.container():
        st.title("Kidney Disease Predictor")
        st.write("This app predicts whether a patient has kidney disease based on various measurements. You can update the measurements using the sliders in the sidebar.")
    
    col1, col2 = st.columns([4,1])
    
    with col1:
        correlation_matrix = get_correlation_matrix(input_data)
        st.plotly_chart(correlation_matrix)
    
    with col2:
        kidney_disease_predictions(input_data)

if __name__ == '__main__':
    main()
