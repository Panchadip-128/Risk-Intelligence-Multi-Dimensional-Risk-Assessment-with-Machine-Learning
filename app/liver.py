from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
import pickle
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import plotly.express as px
import seaborn as sns


def get_clean_data():
    data = pd.read_csv("indian_liver_patient.csv")
    data['Gender'] = data['Gender'].map({'Male': 1, 'Female': 0})
    return data


def liver_disease_sidebar():
    st.sidebar.header("Liver Disease Measurements")
    
    data = get_clean_data()
    
    slider_labels = [
        ("Age", "Age"),
        ("Gender", "Gender"),
        ("Total Bilirubin", "Total_Bilirubin"),
        ("Direct Bilirubin", "Direct_Bilirubin"),
        ("Alkaline Phosphotase", "Alkaline_Phosphotase"),
        ("Alamine Aminotransferase", "Alamine_Aminotransferase"),
        ("Aspartate Aminotransferase", "Aspartate_Aminotransferase"),
        ("Total Proteins", "Total_Protiens"),  # Corrected to "Total_Protiens"
        ("Albumin", "Albumin"),
        ("Albumin and Globulin Ratio", "Albumin_and_Globulin_Ratio")
    ]

    input_dict = {}

    for label, key in slider_labels:
        if key == "Gender":
            input_dict[key] = st.sidebar.selectbox(label, options=[0, 1], format_func=lambda x: 'Female' if x == 0 else 'Male')
        else:
            input_dict[key] = st.sidebar.slider(
                label,
                min_value=float(0),
                max_value=float(data[key].max()),
                value=float(data[key].mean())
            )
    
    return input_dict

def get_scaled_values(input_dict):
    data = get_clean_data()
    
    X = data.drop(['Dataset'], axis=1)
    
    scaled_dict = {}
    
    for key, value in input_dict.items():
        max_val = float(X[key].max())
        min_val = float(X[key].min())
        scaled_value = (value - min_val) / (max_val - min_val)
        scaled_dict[key] = scaled_value
    
    return scaled_dict
def get_scaled_values1(data):
    data=get_clean_data()
    data= data.drop(['Dataset'], axis=1)

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    scaled_df = pd.DataFrame(scaled_data, columns=data.columns)
    return scaled_df

# Function to create radar chart
def plot_distributions():
    data=get_clean_data()
    data1 = data[['Total_Bilirubin', 'Direct_Bilirubin', 'Total_Protiens', 'Albumin', 'Albumin_and_Globulin_Ratio']]
    plt.figure(figsize=(15, 15))
    plotnumber = 1

    for column in data1:
        if plotnumber <= 5:
            ax = plt.subplot(3, 2, plotnumber)
            sns.histplot(data1[column], kde=True, color='tab:purple')
            plt.xlabel(column, fontsize=20, color='white')  # Set xlabel color to white
            plt.ylabel('Frequency', fontsize=20, color='white')  # Set ylabel color to white
            plt.title(f'Distribution of {column}', fontsize=22, color='white')  # Set title color to white
            ax.set_facecolor('none')
            plotnumber += 1
    plt.tight_layout()
    fig = plt.gcf()
    fig.patch.set_facecolor('none')
    fig.patch.set_alpha(0)
    st.pyplot(fig)

def get_radar_chart2():
    # Define the columns to be used in the radar chart
    df=get_clean_data()
    input_data = pd.DataFrame(df)
    categories = ['Total_Bilirubin', 'Direct_Bilirubin', 'Alkaline_Phosphotase', 
                  'Alamine_Aminotransferase', 'Aspartate_Aminotransferase', 'Total_Protiens', 
                  'Albumin', 'Albumin_and_Globulin_Ratio']

    # Scale the input data
    input_data = get_scaled_values1(input_data)
    
    fig = go.Figure()

    # Add a trace for each data row (assuming each row is a separate patient record)
    for index, row in input_data.iterrows():
        fig.add_trace(go.Scatterpolar(
            r=row[categories].values,
            theta=categories,
            fill='toself',
            name=f'Record {index}'
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
def get_radar_chart_liver(input_data):
    input_data = get_scaled_values(input_data)
    
    categories = ['Age','Total_Bilirubin', 'Direct_Bilirubin', 'Alkaline_Phosphotase', 
                 'Alamine_Aminotransferase',"Total_Protiens","Aspartate_Aminotransferase",'Albumin']

    values = [input_data[category] for category in categories]

    fig = px.bar(
        x=categories,
        y=values,
        labels={'x': 'Features', 'y': 'Scaled Values'},
        title="Liver Disease Measurements"
    )
    
    return fig
    
def get_line_chart(input_data):
    data = get_clean_data()

    # Calculate mean and worst values
    mean_values = data.mean()
    worst_values = data.max()

    categories = data.columns.tolist()[:-1]  # Exclude 'Dataset' column
    values_mean = mean_values[categories]
    values_worst = worst_values[categories]
    #se_values = data.std(categories)  # Standard deviation as standard error for this example
    se_values = data.std() 
    values_se = se_values[categories]


    # Assume real-time input values are the last row of the input_data
    real_time_values = input_data

    # Scale the values
    mean_scaled = get_scaled_values1(pd.DataFrame([values_mean])).iloc[0].tolist()
    worst_scaled = get_scaled_values1(pd.DataFrame([values_worst])).iloc[0].tolist()
    real_time_scaled = list(get_scaled_values(real_time_values).values())
    se_scaled = get_scaled_values1(pd.DataFrame([values_se])).iloc[0].tolist()


    fig = px.line(
        x=categories * 3,  # Repeat categories for mean, worst, and real-time
        y=mean_scaled + se_scaled + real_time_scaled,
        color=['Mean'] * len(categories) + ['Standard error'] * len(categories) + ['Real-Time'] * len(categories),
        labels={'x': 'Features', 'y': 'Scaled Values'},
        title="Liver Disease Measurements - Mean, SDE, and Real-Time Values"
    )
    fig.update_layout(
        xaxis={'categoryorder': 'array', 'categoryarray': categories},  # Ensure correct category order
        legend_title='Values',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    return fig

def liver_disease_predictions(input_data):
    model = pickle.load(open("model/liver_model.pkl", "rb"))
    scaler = pickle.load(open("model/liver_scaler.pkl", "rb"))
    
    input_array = np.array(list(input_data.values())).reshape(1, -1)
    
    input_array_scaled = scaler.transform(input_array)
    
    prediction = model.predict(input_array_scaled)
    
    st.subheader("Liver Disease Prediction")
    st.write("The liver disease status is:")
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
        page_title="Liver Disease Predictor",
        page_icon=":liver:",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    with open("assets/style.css") as f:
        st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)
    
    input_data = liver_disease_sidebar()
    
    with st.container():
        st.title("Liver Disease Predictor")
        st.write("This app predicts whether a patient has liver disease based on various measurements. You can update the measurements using the sliders in the sidebar.")
    
    col1, col2 = st.columns([4,1])
    
    with col1:
        radar_chart = get_radar_chart_liver(input_data)
        line_chart=get_line_chart(input_data)
        st.plotly_chart(radar_chart,line_chart)
    with col2:
        liver_disease_predictions(input_data)

if __name__ == '__main__':
    main()
