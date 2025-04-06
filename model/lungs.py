import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from imblearn.over_sampling import ADASYN
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.svm import SVC
import pickle as pickle

def get_clean_data():
    df=pd.read_csv('survey lung cancer.csv')
    df=df.drop_duplicates()
    le=preprocessing.LabelEncoder()
    df['GENDER']=le.fit_transform(df['GENDER'])
    df['LUNG_CANCER']=le.fit_transform(df['LUNG_CANCER'])
    df['SMOKING']=le.fit_transform(df['SMOKING'])
    df['YELLOW_FINGERS']=le.fit_transform(df['YELLOW_FINGERS'])
    df['ANXIETY']=le.fit_transform(df['ANXIETY'])
    df['PEER_PRESSURE']=le.fit_transform(df['PEER_PRESSURE'])
    df['CHRONIC DISEASE']=le.fit_transform(df['CHRONIC DISEASE'])
    df['FATIGUE ']=le.fit_transform(df['FATIGUE '])
    df['ALLERGY ']=le.fit_transform(df['ALLERGY '])
    df['WHEEZING']=le.fit_transform(df['WHEEZING'])
    df['ALCOHOL CONSUMING']=le.fit_transform(df['ALCOHOL CONSUMING'])
    df['COUGHING']=le.fit_transform(df['COUGHING'])
    df['SHORTNESS OF BREATH']=le.fit_transform(df['SHORTNESS OF BREATH'])
    df['SWALLOWING DIFFICULTY']=le.fit_transform(df['SWALLOWING DIFFICULTY'])
    df['CHEST PAIN']=le.fit_transform(df['CHEST PAIN'])
    df['LUNG_CANCER']=le.fit_transform(df['LUNG_CANCER'])
    df_new=df.drop(columns=['GENDER','AGE', 'SMOKING', 'SHORTNESS OF BREATH'])
    #df_new['ANXYELFIN']=df_new['ANXIETY']*df_new['YELLOW_FINGERS']
    return df_new

def create_model(df_new):
   X = df_new.drop('LUNG_CANCER', axis = 1)
   y = df_new['LUNG_CANCER']
   adasyn = ADASYN(random_state=42)
   X, y = adasyn.fit_resample(X, y)
   scaler = StandardScaler()
   X = scaler.fit_transform(X)
   X_train, X_test, y_train, y_test= train_test_split(X, y, test_size= 0.25, random_state=0)
   svc_model = SVC()
   svc_model.fit(X_train, y_train)
   y_svc_pred= svc_model.predict(X_test)
   svc_cr=classification_report(y_test, y_svc_pred)
   print(svc_cr)
   return svc_model,scaler


def main():
  df=get_clean_data()
  model,scaler=create_model(df)
  with open('model/lung_model1.pkl', 'wb') as f:
    pickle.dump(model, f)
  with open('model/lung_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)




if __name__ == '__main__':
  main()