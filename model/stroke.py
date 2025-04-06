import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, accuracy_score, f1_score
import pickle

def get_clean_data(file_path):
    # Load data
    df = pd.read_csv(file_path)
    
    # Drop rows with 'Other' gender
    df.drop(df[df['gender'] == 'Other'].index, inplace=True)
    
    # Drop unnecessary columns (assuming 'id' is not relevant for prediction)
    df = df.drop(columns=['id'])
    
    # Encode categorical variables
    object_cols = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
    label_encoder = LabelEncoder()
    for col in object_cols:
        df[col] = label_encoder.fit_transform(df[col])
    
    return df

def create_model(df):
    # Separate features and target
    X = df.drop(columns=['stroke'])
    y = df['stroke']
    
    # Handle missing values (using most frequent strategy for simplicity)
    imputer = SimpleImputer(strategy='most_frequent')
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    
    # Scale numerical features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Handle class imbalance using SMOTE
    sampler = SMOTE(random_state=42)
    X_resampled, y_resampled = sampler.fit_resample(X, y)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
    
    # Initialize and train a SVM classifier with probability enabled
    model = SVC(kernel='rbf', probability=True, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate model performance
    y_pred = model.predict(X_test)
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Accuracy Score:", accuracy_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))
    
    return model, scaler

def save_model(model, scaler, model_file='stroke1_model.pkl', scaler_file='stroke1_scaler.pkl'):
    # Save the trained model and scaler
    with open(model_file, 'wb') as file:
        pickle.dump(model, file)
    with open(scaler_file, 'wb') as file:
        pickle.dump(scaler, file)
    
    print(f"Model saved as '{model_file}' and scaler saved as '{scaler_file}'.")

def main():
    # File path for your dataset
    file_path = 'healthcare-dataset-stroke-data.csv'
    
    # Clean and preprocess the data
    df = get_clean_data(file_path)
    
    # Create and train the model
    model, scaler = create_model(df)
    
    # Save the model and scaler
    save_model(model, scaler)

if __name__ == '__main__':
    main()
