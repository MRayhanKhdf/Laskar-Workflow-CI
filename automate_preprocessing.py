import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

def load_and_preprocess(filepath):
    df = pd.read_csv("Dataset/ai_job_dataset.csv")

    # Drop kolom tidak relevan atau sulit diproses
    df = df.drop(columns=[
        'job_id', 'salary_currency', 'posting_date',
        'application_deadline', 'company_name', 'required_skills'
    ])
    
    df = df.dropna()

    # Encode kolom kategorikal
    categorical_cols = df.select_dtypes(include='object').columns
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    # Split fitur dan target
    X = df.drop(columns=['salary_usd'])
    y = df['salary_usd']

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test, scaler, X.columns