import pandas as pd
import streamlit as st


from breast_cancer import add_sidebar_breast,get_radar_chart,add_predictions,get_line_chart_breast,plot_diagnosis_pie_chart
from liver import liver_disease_sidebar,liver_disease_predictions,get_radar_chart_liver,get_line_chart,plot_distributions
from heart import add_sidebar_heart,heart_disease_predictions,get_radar_chart_heart,plot_line_chart_heart
from diabetes import diabetes_sidebar,diabetes_predictions,get_radar_chart_diabetes,get_parallel_coordinates_plot,get_line_chart_diabetes,display_correlation_matrix,get_clean_data
from kidney import kidney_disease_sidebar,kidney_disease_predictions,get_correlation_matrix
from stroke import stroke_disease_predictions,stroke_disease_sidebar,get_bar_chart,get_clean_data,pie_chart,generate_pie_charts,get_line_chart_stroke
def add_sidebar():
    st.sidebar.header("Select a Prediction Model")
    model_choice = st.sidebar.selectbox(
        "Choose a model:",
        ("Breast Cancer Prediction", "Liver Disease Prediction", "Stroke Prediction", 
         "Diabetes Prediction", "Heart Disease Prediction")
    )
    return model_choice

def main():
  st.set_page_config(
    page_title="Multi_Disease_Predictor",
    page_icon=":female-doctor:",
    layout="wide",
    initial_sidebar_state="expanded"
  )
  with open("assets/style.css") as f:
    st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)
  model_choice = add_sidebar()
  if model_choice == "Breast Cancer Prediction":
        input_data = add_sidebar_breast()
        st.title("Breast Cancer Predictor")
        st.info("Our Breast Cancer Prediction Model leverages advanced machine learning to analyze key features such as age, tumor size, and texture, providing a quick and accurate risk assessment for breast cancer. This tool is designed to aid in early detection and prompt treatment, offering reliable results through a user-friendly interface that allows patients and healthcare providers to input relevant medical and demographic information and receive immediate feedback on potential breast cancer risk. ")
        col1, col2 = st.columns([4,1])
        with col1:
            with st.container():
                col3,col4=st.columns(2)
                with col3:
                    radar_chart = get_radar_chart(input_data)
                    st.plotly_chart(radar_chart)
                    st.image('assets/__results___40_0 (1).png',width=400)

                with col4:
                     plot_diagnosis_pie_chart()
                     st.image('assets/__results___47_0.png')
        with col2:
            add_predictions(input_data)
            
        with st.container():
                    line_chart = get_line_chart_breast(input_data)
                    st.plotly_chart(line_chart)
        report_data = {
    'Label': ['0', '1', 'accuracy', 'macro avg', 'weighted avg'],
    'precision': [0.97, 0.93, None, 0.95, 0.96],
    'recall': [0.96, 0.95, None, 0.96, 0.96],
    'f1-score': [0.96, 0.94, 0.96, 0.95, 0.96],
    'support': [71, 43, 114, 114, 114]
}

        # Creating a DataFrame for the report
        report_df = pd.DataFrame(report_data)

        # Adding Test Score and Train Score to the Streamlit app
        st.title('Model Performance Report')

        # Display Train and Test Scores
        st.subheader('Scores')
        st.info(f"**Test Score:** 0.956")
        st.info(f"**Train Score:** 1.000")

        # Display Classification Report
        st.subheader('Classification Report')
        st.table(report_df)

  elif model_choice=="Liver Disease Prediction":
        input_data = liver_disease_sidebar()
        st.title("Liver Disease Predictor")
        st.info(" Liver Disease Prediction ML model uses patient-specific features such as Age, Gender, and various liver enzyme levels to predict the likelihood of liver disease. By analyzing these features, the model aids in early detection and treatment, improving clinical decision-making and patient outcomes. Advanced algorithms like decision trees, random forests, or neural networks ensure high accuracy and reliability in predictions.")
        col1, col2 = st.columns([4,1])
        with col1:
            with st.container():
                col3,col4=st.columns([2,3])
                with col3:
                    radar_chart = get_radar_chart_liver(input_data)
                    line_chart=get_line_chart(input_data)
                    st.plotly_chart(radar_chart)
                    st.info('Male vs Female')
                    st.image('assets/__results___8_1.png')
                    
                with col4:
                    plot_distributions()
                    st.image('assets/__results___9_0.png')

        with col2:
            liver_disease_predictions(input_data)
        st.plotly_chart(line_chart)
        classification_report_data = {
    'precision': [0.66, 0.69],
    'recall': [0.60, 0.74],
    'f1-score': [0.63, 0.71],
    'support': [93, 110]
}

# Create a DataFrame from the classification report data
        df_classification_report = pd.DataFrame(classification_report_data, index=[1, 2])

# Streamlit app
        st.title('Classification Report')
        st.table(df_classification_report)

  elif model_choice=="Heart Disease Prediction":
      with st.container():
        st.title("Heart Disease Predictor")
        st.info("In this Heart disease predictor, we delve into a dataset encapsulating various health metrics from heart patients, including age, blood pressure, heart rate, and more. Our goal is to develop a predictive model capable of accurately identifying individuals with heart disease. Given the grave implications of missing a positive diagnosis, our primary emphasis is on ensuring that the model identifies all potential patients, making recall for the positive class a crucial metric.")
      input_data=add_sidebar_heart()
      col1, col2 = st.columns([4,1])
      with col1:
          col3,col4=st.columns(2)
          with col3:    
            fig=get_radar_chart_heart(input_data)
            st.plotly_chart(fig)
            st.image('assets/__results___15_0.png')
          with col4:
            st.image('assets/__results___23_0.png')
            st.image('assets/__results___30_1.png')
      with col2:
          heart_disease_predictions(input_data)
      st.image('assets/__results___13_0.png')
      fig1=plot_line_chart_heart(input_data) 
      st.plotly_chart(fig1)
      classification_report = {
    'precision': [0.94, 0.73],
    'recall': [0.57, 0.97],
    'f1-score': [0.71, 0.83],
    'support': [28, 33]
}

# Create a DataFrame from the classification report data
      df = pd.DataFrame(classification_report, index=['0', '1'])

    # Add accuracy and macro/weighted averages if needed
      accuracy = {'precision': '-', 'recall': '-', 'f1-score': '-', 'support': 61}
      macro_avg = {'precision': 0.83, 'recall': 0.77, 'f1-score': 0.77, 'support': 61}
      weighted_avg = {'precision': 0.83, 'recall': 0.79, 'f1-score': 0.78, 'support': 61}

    # Display classification report in a Streamlit table
      st.header("Classification Report")
      st.table(df)

      st.subheader("Additional Metrics")
      st.table([accuracy, macro_avg, weighted_avg])
      
  elif model_choice=="Diabetes Prediction":
      st.title("Diabetes Predictor")

      st.info("diabetes prediction model utilizes machine learning algorithms to analyze key health indicators such as glucose levels, BMI, age, and lifestyle factors like smoking. By processing these data points, the model provides accurate predictions about an individual's likelihood of developing diabetes. This tool not only aids in early detection but also empowers healthcare professionals to implement preventive measures, thereby improving patient outcomes and reducing the burden of diabetes-related complications.")
      input_data=diabetes_sidebar()
      col1, col2 = st.columns([4,1])
      with col1:
          with st.container():
              col3,col4=st.columns([2,3])
              with col3:
                  radar_chart = get_radar_chart_diabetes(input_data)
                  st.plotly_chart(radar_chart)
                  st.image('assets/__results___17_1.png')
              with col4:
                  data=get_clean_data()
                  fig1=get_parallel_coordinates_plot(input_data)
                  st.plotly_chart(fig1)
                  st.image('assets/__results___29_1.png')
      with col2:
          diabetes_predictions(input_data)
      
      st.image('assets/__results___54_0-removebg-preview.png')
      line_chart=get_line_chart_diabetes(input_data)  
      st.plotly_chart(line_chart) 
      data = {
    'Class': ['0.0', '1.0', 'accuracy', 'macro avg', 'weighted avg'],
    'precision': [0.86, 0.65, '-', 0.76, 0.79],
    'recall': [0.84, 0.68, '-', 0.76, 0.79],
    'f1-score': [0.85, 0.67, '-', 0.76, 0.79],
    'support': [107, 47, 154, 154, 154]
}

# Create a DataFrame from the data
      df = pd.DataFrame(data)
      st.title('Model Performance Report')

        # Display Train and Test Scores
      st.subheader('Scores')
# Display the classification report in a tabular form
      st.table(df)
  elif model_choice=="Stroke Prediction": 
      st.title("Stroke Predictor")
      st.info("This app employs a machine learning model to predict the likelihood of stroke based on key health indicators such as age, gender, hypertension, heart disease, average glucose level, BMI, and smoking status. By analyzing these factors, the model provides insights into the probability of stroke occurrence, aiding healthcare professionals in early intervention and personalized patient care. Please note, this app serves for demonstration purposes and does not substitute professional medical advice.")
    
      input_data=stroke_disease_sidebar()
      input_datat2=add_sidebar_breast()
      df=get_clean_data()
      col1,col2=st.columns([4,1])
      with col1:
        work_type_fig, gender_fig, smoking_status_fig, residence_type_fig = generate_pie_charts(df)
        with st.container():
            col3,col4=st.columns(2)
            with col3:
                st.plotly_chart(work_type_fig)
                st.image('assets/__results___24_1.png')
                st.image('assets/__results___27_1.png')
            with col4:
                 st.plotly_chart(smoking_status_fig)
                 st.image('assets/__results___30_1.png')
                 st.image('assets/__results___28_1.png')
      with col2:
          stroke_disease_predictions(input_data)
      fig=get_line_chart_breast(input_datat2)
      st.plotly_chart(fig)
      classification_report_data = {
    "": ["precision", "recall", "f1-score", "support"],
    "0": [0.96, 0.97, 0.96, 1194],
    "1": [0.15, 0.13, 0.14, 52],
    "accuracy": ["", "", 0.93, 1246],
    "macro avg": [0.56, 0.55, 0.55, 1246],
    "weighted avg": [0.93, 0.93, 0.93, 1246]
}

# Convert data to DataFrame
      df_classification_report = pd.DataFrame(classification_report_data)

# Display as a table in Streamlit
      st.title('Model Performance Report')

        # Display Train and Test Scores
      st.subheader('Scores')
      st.table(df_classification_report)
if __name__ == '__main__':
  main()

