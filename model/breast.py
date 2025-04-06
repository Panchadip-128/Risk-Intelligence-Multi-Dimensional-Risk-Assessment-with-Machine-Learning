import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
import pickle as pickle


def create_model(data): 
  X = data.drop(['diagnosis'], axis=1)
  y = data['diagnosis']
  
  scaler = StandardScaler()
  X = scaler.fit_transform(X)
  
  X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
  )
  model = GradientBoostingClassifier()
  model = GradientBoostingClassifier(learning_rate = 1, loss = 'exponential', n_estimators = 200)
  model.fit(X_train, y_train)
  y_pred = model.predict(X_test)
  print('Accuracy of our model: ', accuracy_score(y_test, y_pred))
  print("Classification report: \n", classification_report(y_test, y_pred))
  report = classification_report(y_test, y_pred, target_names=data.target_names, output_dict=True)

  
  return model, scaler,report

def get_clean_data():
  data = pd.read_csv("data.csv")
  data = data.drop(['Unnamed: 32', 'id'], axis=1)
  data['diagnosis'] = data['diagnosis'].map({ 'M': 1, 'B': 0 })
  return data

def main():
  data = get_clean_data()
  model, scaler = create_model(data)
  with open('model/model.pkl', 'wb') as f:
    pickle.dump(model, f)
  with open('model/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
  

if __name__ == '__main__':
  main()