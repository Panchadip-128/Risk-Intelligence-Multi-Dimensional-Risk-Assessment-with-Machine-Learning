import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pickle

# Function to clean and preprocess data
def get_clean_data():
    df = pd.read_csv('survey lung cancer.csv')
    # Encoding categorical variables if needed
    # Example: df['GENDER'] = df['GENDER'].map({'M': 1, 'F': 0})
    return df

# Sidebar for input
def lung_disease_sidebar():
    st.sidebar.header("Lung Disease Measurements")
    
    data = get_clean_data()
    
    slider_labels = [
        ("Age", "AGE"),
        ("Gender", "GENDER"),
        ("Smoking", "SMOKING"),
        ("Yellow Fingers", "YELLOW_FINGERS"),
        ("Anxiety", "ANXIETY"),
        ("Peer Pressure", "PEER_PRESSURE"),
        ("Chronic Disease", "CHRONIC DISEASE"),
        ("FATIGUE", "FATIGUE"),
        ("Allergy", "ALLERGY"),
        ("Wheezing", "WHEEZING"),
        ("Alcohol Consuming", "ALCOHOL CONSUMING"),
        ("Coughing", "COUGHING"),
        ("Shortness of Breath", "SHORTNESS OF BREATH"),
        ("Swallowing Difficulty", "SWALLOWING DIFFICULTY"),
        ("Chest Pain", "CHEST PAIN"),
    ]

    input_dict = {}

    for label, key in slider_labels:
        if key == "GENDER":
            input_dict[key] = st.sidebar.selectbox(label, options=[0, 1], format_func=lambda x: 'Female' if x == 0 else 'Male')
        elif key == "AGE":
            input_dict[key] = st.sidebar.slider(
                label,
                min_value=float(0),
                max_value=float(data[key].max()),
                value=float(data[key].mean())
            )
        else:
            input_dict[key] = st.sidebar.slider(
            label,
            min_value=1,
            max_value=2,
            value=1
        )
    
    return input_dict

# Scaling input data
def get_scaled_values(input_dict):
    data = get_clean_data()
    data=data.drop(['LUNG_CANCER'],axis=1)
    
    scaled_dict = {}
    
    for key, value in input_dict.items():
        max_val = 2
        min_val = 1
        scaled_value = (value - min_val) / (max_val - min_val)
        scaled_dict[key] = scaled_value
    
    return scaled_dict

# Radar chart visualization
def get_radar_chart_lung(input_data):
    categories = list(input_data.keys())
    values = list(input_data.values())
    
    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Input Values'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True
    )
    
    return fig

# Function for lung disease prediction
def lung_disease_predictions(input_data):
    model = pickle.load(open("model/lung_model.pkl", "rb"))  # Load your trained model
    scaler = pickle.load(open("model/lung_scaler.pkl", "rb"))  # Load scaler if used
    
    input_array = np.array(list(input_data.values())).reshape(1, -1)
    
    input_array_scaled = scaler.transform(input_array)  # Scale input data if needed
    
    prediction = model.predict(input_array_scaled)
    predicted_prob_disease = model.predict_proba(input_array_scaled)[0][1]
    
    st.subheader("Lung Disease Prediction")
    st.write("Prediction Result:")
    
    if prediction[0] == 1:
        st.write("<span class='diagnosis malicious'>Disease Present</span>", unsafe_allow_html=True)
    else:
        st.write("<span class='diagnosis benign'>No Disease</span>", unsafe_allow_html=True)
    
    st.write("Probability of no disease: ", 1 - predicted_prob_disease)
    st.write("Probability of disease: ", predicted_prob_disease)
    
    st.write("This app can assist medical professionals in making a diagnosis, but should not be used as a substitute for a professional diagnosis.")

# Main function to run the Streamlit app
def main():
    st.set_page_config(
        page_title="Lung Disease Predictor",
        page_icon=":lungs:",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    with open("assets/style.css") as f:
        st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)
    
    input_data = lung_disease_sidebar()
    
    with st.container():
        st.title("Lung Disease Predictor")
        st.write("This app predicts the presence of lung disease based on various measurements. Adjust the sliders in the sidebar to input patient data.")
    
    col1, col2 = st.columns([4,1])
    
    with col1:
        radar_chart = get_radar_chart_lung(input_data)
        st.plotly_chart(radar_chart)
    with col2:
        lung_disease_predictions(input_data)

if __name__ == '__main__':
    main()
