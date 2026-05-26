import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

CONTAMINATION = 0.07
RANDOM_STATE = 42

models = [
    ('IsolationForest', IsolationForest(contamination=CONTAMINATION, random_state=RANDOM_STATE)),
    ('OneClassSVM', OneClassSVM(kernel='rbf', gamma='scale', nu=CONTAMINATION)),
    ('LocalOutlierFactor', LocalOutlierFactor(n_neighbors=20, contamination=CONTAMINATION, novelty=True)),
    ('EllipticEnvelope', EllipticEnvelope(contamination=CONTAMINATION, random_state=RANDOM_STATE))
]

def plot_group(group_name):
    X_train = pd.read_csv(f'X_train_{group_name}.csv')
    X_test = pd.read_csv(f'X_test_{group_name}.csv')

    pca2 = PCA(n_components=2).fit(X_train)
    X_train_2 = pca2.transform(X_train)
    X_test_2 = pca2.transform(X_test)

    for j in range(2):
        if X_train_2[:, j].mean() < 0:
            X_train_2[:, j] *= -1
            X_test_2[:, j] *= -1

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for ax, (name, model) in zip(axes, models):
        model.fit(X_train)
        preds = model.predict(X_test)
        labels = (preds == -1).astype(int)

        n_outliers = int(labels.sum())
        colors = np.where(labels == 1, 'red', 'blue')

        ax.scatter(X_test_2[:, 0], X_test_2[:, 1], c=colors, s=12, alpha=0.7)
        ax.set_title(f"{name} — outliers {n_outliers}/{len(labels)}")
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.grid(True)

    red_patch = mpatches.Patch(color='red', label='outlier')
    blue_patch = mpatches.Patch(color='blue', label='inlier')
    fig.legend(handles=[red_patch, blue_patch], loc='upper right')
    fig.suptitle(f"Anomaly detection - {group_name.upper()} wine")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

plot_group('red')
plot_group('white')

def clean_with_isolation_forest(group_name):
    X_train = pd.read_csv(f'X_train_{group_name}.csv')
    X_test = pd.read_csv(f'X_test_{group_name}.csv')
    y_train = pd.read_csv(f'y_train_{group_name}.csv')
    y_test = pd.read_csv(f'y_test_{group_name}.csv')

    iso = IsolationForest(contamination=CONTAMINATION, random_state=RANDOM_STATE)
    train_pred = iso.fit_predict(X_train)

    train_mask = train_pred == 1
    X_train_clean = X_train.loc[train_mask].reset_index(drop=True)
    y_train_clean = y_train.loc[train_mask].reset_index(drop=True)

    print(f"\n{group_name.upper()} cleaned train:")
    print(f"before: {X_train.shape[0]}")
    print(f"after:  {X_train_clean.shape[0]}")
    print(f"removed: {(~train_mask).sum()}")

    X_train_clean.to_csv(f'X_train_{group_name}_clean.csv', index=False)
    y_train_clean.to_csv(f'y_train_{group_name}_clean.csv', index=False)

    X_test.to_csv(f'X_test_{group_name}_clean.csv', index=False)
    y_test.to_csv(f'y_test_{group_name}_clean.csv', index=False)

    return X_train_clean, y_train_clean, X_test, y_test

clean_with_isolation_forest('red')
clean_with_isolation_forest('white')