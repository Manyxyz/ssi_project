import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from joblib import dump
import os

def prepare_group(csv_path, prefix):
    df = pd.read_csv(csv_path, sep=';')
    df.columns = [c.strip().lower().replace(' ', '_').replace('"', '') for c in df.columns]

    print(f"\n{prefix.upper()} shape: {df.shape}")
    print(f"{prefix.upper()} nulls:\n{df.isnull().sum()}")
    print(f"{prefix.upper()} duplicates: {df.duplicated().sum()}")

    df = df.drop_duplicates()

    X = df.drop('quality', axis=1)
    y = df['quality']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler().fit(X_train)
    X_train_scaled = pd.DataFrame(
        scaler.transform(X_train),
        columns=X.columns,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X.columns,
        index=X_test.index
    )

    pca = PCA(n_components=0.95)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    print(f"{prefix.upper()} train shape: {X_train_scaled.shape}")
    print(f"{prefix.upper()} test shape: {X_test_scaled.shape}")
    print(f"{prefix.upper()} PCA: {X.shape[1]} features -> {X_train_pca.shape[1]} components")
    print(f"{prefix.upper()} explained variance: {pca.explained_variance_ratio_.sum():.4f}")

    X_train_scaled.to_csv(f'X_train_{prefix}.csv', index=False)
    X_test_scaled.to_csv(f'X_test_{prefix}.csv', index=False)
    y_train.to_csv(f'y_train_{prefix}.csv', index=False)
    y_test.to_csv(f'y_test_{prefix}.csv', index=False)

    os.makedirs('models', exist_ok=True)
    dump(scaler, f'models/scaler_{prefix}.joblib')

    print(f"{prefix.upper()} data prepared and saved")

prepare_group('wine+quality/winequality-red.csv', 'red')
prepare_group('wine+quality/winequality-white.csv', 'white')

print("\nAll data prepared and saved")