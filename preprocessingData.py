import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocessing(df):
    if df.isnull().sum().sum() > 0:
        print("\nThere are null entries in the dataset.")
        print(df.isnull().sum())
        df = df.dropna()
        print("\nAfter :")
        print(df.isnull().sum())

    scaler = StandardScaler()
    df = pd.get_dummies(df)
    df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    print(df.describe())
    return df