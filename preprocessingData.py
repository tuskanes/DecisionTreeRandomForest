import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocessing(df):
    if df.isnull().sum().sum() > 0:
        print("\nThere are null entries in the dataset.")
        print(df.isnull().sum())
        df = df.dropna(subset=['Accident'])
        df = df.fillna(df.mean(numeric_only=True))
        print("\nAfter :")
        print(df.isnull().sum())

    if df.duplicated().sum() > 0 :
        print("\nThere are duplicate entries in the dataset.")
        print(df.duplicated().sum())
        df = df[~df.duplicated()]
        print("\nAfter :")
        print(df.duplicated().sum())
    scaler = StandardScaler()
    df = pd.get_dummies(df)
    df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    print(df.describe())
    df['Accident'] = df['Accident'].astype(int)
    return df