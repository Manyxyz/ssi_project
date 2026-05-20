import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_and_merge(red_path, white_path):
    red = pd.read_csv(red_path, sep=';')
    white = pd.read_csv(white_path, sep=';')
    red['color'] = 'red'
    white['color'] = 'white'
    df = pd.concat([red, white], ignore_index=True)
    df.columns = [c.strip().lower().replace(' ', '_').replace('"','') for c in df.columns]
    return df

df = load_and_merge('wine+quality/winequality-red.csv', 'wine+quality/winequality-white.csv')
print(df.shape)
print(df['color'].value_counts())
print(df.isnull().sum())