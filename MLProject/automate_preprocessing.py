import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer

def load_and_preprocess(filepath):
    # Load dataset
    df = pd.read_csv(filepath)

    # Drop kolom yang tidak berguna untuk model
    df.drop(columns=["job_id", "posting_date", "application_deadline", "company_name"], inplace=True)

    # Definisikan fitur dan target
    X = df.drop("salary_usd", axis=1)
    y = df["salary_usd"]

    # Tentukan kolom numerik dan kategorikal
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    # Pipeline numerik
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    # Pipeline kategorikal
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Gabungkan preprocessing
    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features)
    ])

    # Preprocessing data
    X_processed = preprocessor.fit_transform(X)

    # Split train-test
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test, preprocessor, df
