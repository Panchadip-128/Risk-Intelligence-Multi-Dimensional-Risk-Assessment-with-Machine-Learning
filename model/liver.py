import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import KFold, cross_val_score, train_test_split


def get_clean_data():
  patients = pd.read_csv("indian_liver_patient.csv")
  patients['Gender']=patients['Gender'].apply(lambda x:1 if x=='Male' else 0)
  patients['Albumin_and_Globulin_Ratio'].mean()
  patients=patients.fillna(0.94)
  return patients

def create_model(patients):
    X=patients[['Age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin',
       'Alkaline_Phosphotase', 'Alamine_Aminotransferase',
       'Aspartate_Aminotransferase', 'Total_Protiens', 'Albumin',
       'Albumin_and_Globulin_Ratio']]
    y=patients['Dataset']
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)
    logmodel = LogisticRegression(C=1, penalty='l1', solver='liblinear')
    logmodel.fit(X_train, y_train)
    y_pred = logmodel.predict(X_test)
    print('Accuracy of our model: ', accuracy_score(y_test, y_pred))
    print("Classification report: \n", classification_report(y_test, y_pred))
    return logmodel,scaler
def main():
 df=get_clean_data()
 model,scaler=create_model(df)
 with open('model/liver_model.pkl', 'wb') as f:
    pickle.dump(model, f)
 with open('model/liver_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

if __name__ == '__main__':
  main()




