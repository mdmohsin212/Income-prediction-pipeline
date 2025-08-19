import pandas as pd
import os
from sklearn.model_selection import train_test_split

df = pd.read_csv('./data/raw/raw_data.csv')

def data_preprocess(df : pd.DataFrame) -> pd.DataFrame:
    try:
        df['workclass'] = df['workclass'].fillna(df['workclass'].mode()[0])
        df['occupation'] = df['occupation'].fillna(df['occupation'].mode()[0])
        df['native-country'] = df['native-country'].fillna(df['native-country'].mode()[0])
        return df
    
    except KeyError as e:
        print(f'Error : Missing Column {e} in Dataframe')
        raise

def encode_data(df : pd.DataFrame) -> pd.DataFrame:
    try:
        df = pd.get_dummies(df, columns=['workclass',
            'education',
            'marital-status',
            'occupation',
            'relationship',
            'race',
            'sex',
            'native-country'], drop_first=True, dtype=int)
        return df
    
    except Exception as e:
        print(f'An unexpected error occurred {e} while saving the data.')
        raise

process_data = data_preprocess(df)
encoded = encode_data(process_data)

train_data, test_data = train_test_split(encoded, test_size=.25, random_state=42)

path = os.path.join("data", "process")
os.makedirs(path, exist_ok=True)

train_data.to_csv(os.path.join(path, 'Train_process.csv'), index=False)
test_data.to_csv(os.path.join(path, 'Test_process.csv'), index=False)