import pickle
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression

def get_clean_data():
    df=pd.read_csv('diabetes.csv')
    df=df.drop_duplicates()
    dataset_new = df
    dataset_new[["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]] = dataset_new[["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]].replace(0, np.nan) 
    dataset_new["Glucose"].fillna(dataset_new["Glucose"].mean(), inplace = True)
    dataset_new["BloodPressure"].fillna(dataset_new["BloodPressure"].mean(), inplace = True)
    dataset_new["SkinThickness"].fillna(dataset_new["SkinThickness"].mean(), inplace = True)
    dataset_new["Insulin"].fillna(dataset_new["Insulin"].mean(), inplace = True)
    dataset_new["BMI"].fillna(dataset_new["BMI"].mean(), inplace = True) 
    return dataset_new


def create_model(dataset_new):
    target_name='Outcome'
    y= dataset_new[target_name]
    X=dataset_new.drop(target_name,axis=1)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.20, random_state = 42, stratify = dataset_new['Outcome'] )
    model = LogisticRegression()
    model.fit(X_train, Y_train)
    y_predict = model.predict(X_test)
    print('Accuracy of our model: ', accuracy_score(Y_test, y_predict))
    print("Classification report: \n", classification_report(Y_test, y_predict))
    return model,scaler


def main():
  df=get_clean_data()
  model,scaler=create_model(df)
  with open('model/diabetes_model.pkl', 'wb') as f:
    pickle.dump(model, f)
  with open('model/diabetes_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

if __name__ == '__main__':
  main()