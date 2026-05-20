import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

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

#duplikaty
print(f"\nDuplikates: {df.duplicated().sum()}")
df = df.drop_duplicates()

def remove_outliers(data, threshold=1.5):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    mask = (data >= lower_bound) & (data <= upper_bound)
    return mask.all(axis=1)

numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
numeric_cols.remove('quality')  
outlier_mask = remove_outliers(df[numeric_cols], threshold=1.5)
print(f"Outliers removed: {(~outlier_mask).sum()}")
df = df[outlier_mask]

#skalowanie
df_encoded = df.copy()
df_encoded['color'] = df_encoded['color'].map({'red': 0, 'white': 1})

X = df_encoded.drop('quality', axis=1)
y = df_encoded['quality']

#standaryzacja
scaler_std = StandardScaler()
X_scaled = scaler_std.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

#SPLIT TRAIN/TEST
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nTrain shape: {X_train.shape}, Test shape: {X_test.shape}")

#PCA 
pca = PCA(n_components=0.95)  #95% wariancji
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
print(f"PCA: {X.shape[1]} features -> {X_train_pca.shape[1]} components")
print(f"Explained variance: {pca.explained_variance_ratio_.sum():.4f}")

#zapis
X_train.to_csv('X_train.csv', index=False)
X_test.to_csv('X_test.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
y_test.to_csv('y_test.csv', index=False)

print("Data prepared and saved")
