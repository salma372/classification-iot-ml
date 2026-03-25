import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import numpy as np

def read_csv_file(path):
    df=pd.read_csv(path)
    return df

def prepare_data(df):
    X=df.drop("Label", axis=1)
    y=df["Label"]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def train_data(X_train, y_train):
    model =  RandomForestClassifier(n_estimators=100, class_weight="balanced")
    model.fit(X_train, y_train)#fct fit entraine le model
    return model

def evaluate_model(model, X_test, y_test):
    y_pred =model.predict(X_test)
    print (classification_report(y_test, y_pred, zero_division=0))

def save_model(model, path="../src/main.pkl"):
    joblib.dump(model, path)
    print(f"\n Modèle sauvegardé dans : {path}")



#fct utile dans ce projet car le dataset contient de trop grande valeurs 


def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(axis=1)
    df = df[indices_to_keep].reset_index(drop=True)  
    
    feature_cols = [col for col in df.columns if col != "Label"]
    df[feature_cols] = df[feature_cols].astype(np.float64)
    
    return df


def main():
    path ="../data/Merged01.csv"
    df =read_csv_file(path)
    df = clean_dataset(df)
    X_train, X_test, y_train, y_test=prepare_data(df)
    model = train_data(X_train, y_train)
    evaluate_model(model,X_test,y_test)
    save_model(model)

if __name__ == "__main__":
    main()