import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, train_test_split


def create_clean_data():
    data=pd.read_csv('kidney_disease.csv')
    data['classification'] = data['classification'].map({'ckd':1,'notckd':0 , 'ckd\t':1})
    data['htn'] = data['htn'].map({'yes':1,'no':0})
    data['dm'] = data['dm'].map({'yes':1,'no':0})
    data['cad'] = data['cad'].map({'yes':1,'no':0})
    data['appet'] = data['appet'].map({'good':1,'poor':0})
    data['ane'] = data['ane'].map({'yes':1,'no':0})
    data['pe'] = data['pe'].map({'yes':1,'no':0})
    data['ba'] = data['ba'].map({'present':1,'notpresent':0})
    data['pcc'] = data['pcc'].map({'present':1,'notpresent':0})
    data['pc'] = data['pc'].map({'abnormal':1,'normal':0})
    data['rbc'] = data['rbc'].map({'abnormal':1,'normal':0})
    data['classification'].value_counts()
    data = data.drop(["pcv","wc","rc","id"], axis = 1)  
    data['age']=data['age'].fillna(data['age'].mean())
    data=data.fillna(data.median())
    n_cols = {'bp':'Blood_Pressure',
            'sg':  'Specific_Gravity','al': 'Albumin','su' : 'Sugar','bgr': 'Blood_Glucose_Random','bu' : 'Blood_Urea' ,
            'sc' : 'Serum_Creatinine','sod' : 'Sodium','pot' : 'Potassium','hemo' : 'Hemoglobin',
            'rbc' : 'Red_Blood_Cells','pc' : 'Pus_Cell','pcc' : 'Pus_Cell_Clumps','ba' : 'Bacteria','htn' : 'Hypertension', 
            'dm' : 'Diabetes_Mellitus','cad' : 'Coronary_Artery_Disease','appet' : 'Appetite','pe' : 'Pedal_Edema',
            'ane' : 'Anemia','classification' : 'Target'}

    data.rename(columns=n_cols ,inplace=True)
    return data

def create_model(data):
    X = data.drop(["Target"], axis = 1) 
    y = data["Target"]
    X = X.drop(["age","Red_Blood_Cells","Pus_Cell_Clumps",
               "Serum_Creatinine","Potassium","Coronary_Artery_Disease",
               "Bacteria","Potassium","Coronary_Artery_Disease"], axis = 1)  
    scaler = StandardScaler()
    X = scaler.fit_transform(X) 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42) 
    clf_rnf=RandomForestClassifier()
    parametrs_rnf={'n_estimators':[3,5], 'max_depth':[2,3]}
    grid_forest=GridSearchCV(clf_rnf, parametrs_rnf, cv=6, n_jobs=-1)
    grid_forest.fit(X_train,y_train)
    best_model_rnf=grid_forest.best_estimator_
    y_pred_rnf=best_model_rnf.predict(X_test)
    ac_rnf = accuracy_score(y_test, y_pred_rnf)
    cr_rnf = classification_report(y_test, y_pred_rnf)
    print("Accuracy score for model " f'{best_model_rnf} : ',ac_rnf)
    print("classification_report for model " f'{best_model_rnf} : \n',cr_rnf)
    return best_model_rnf,scaler
def main():
   df=create_clean_data()
   model ,scaler=create_model(df)
   with open('model/kidney_model.pkl', 'wb') as f:
        pickle.dump(model, f)
   with open('model/kidney_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
  



if __name__ == '__main__':
  main()