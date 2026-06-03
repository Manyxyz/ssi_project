import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, log_loss
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.preprocessing import label_binarize
from sklearn.utils.class_weight import compute_class_weight
from joblib import dump

def load_wine_data(wine_type):
    X_train = pd.read_csv(f"X_train_{wine_type}_clean.csv")
    X_test = pd.read_csv(f"X_test_{wine_type}_clean.csv")
    y_train = pd.read_csv(f"y_train_{wine_type}_clean.csv")
    y_test = pd.read_csv(f"y_test_{wine_type}_clean.csv")

    y_train_3 = y_train["quality"].apply(map_quality)
    y_test_3 = y_test["quality"].apply(map_quality)

    return X_train, X_test, y_train_3, y_test_3


def plot_calibration(y_true, proba, name, ax):
    frac_pos, mean_pred = calibration_curve(y_true, proba, n_bins=10)
    ax.plot(mean_pred, frac_pos, "s-", label=name)


def map_quality(q):
    if q <= 4:
        return 0  # slabe (3-4)
    if q <= 6:
        return 1  # srednie (5-6)
    return 2      # dobre (7-9)


def process_and_evaluate(wine_type):
    print(f"\n{'='*50}")
    print(f"--- ROZPOCZYNAM ANALIZE DLA WINA: {wine_type.upper()} ---")
    print(f"{'='*50}")

    X_train, X_test, y_train, y_test = load_wine_data(wine_type)
    print(f"Dane treningowe: {X_train.shape}, Dane testowe: {X_test.shape}\n")

    classes = np.sort(y_train.unique())
    cw = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)

    alpha = 0.2  # 0.0 = brak wag, 1.0 = pelne balanced
    class_weight = {c: 1.0 + alpha * (w - 1.0) for c, w in zip(classes, cw)}
    sample_weight = y_train.map(class_weight).to_numpy()

    print("[1/2] Trening MLP i GridSearch...")

    base_mlp = MLPClassifier(
        random_state=42,
        max_iter=500,
        early_stopping=True,
        n_iter_no_change=10
    )

    param_grid = {
        "hidden_layer_sizes": [(64, 32, 16), (128, 64, 32), (64, 64, 32)],
        "alpha": [0.0001, 0.001],
        "learning_rate_init": [0.001, 0.0005],
        "activation": ["relu", "tanh"]
    }

    grid = GridSearchCV(
        base_mlp,
        param_grid,
        cv=3,
        n_jobs=-1,
        scoring="f1_macro"
    )
    grid.fit(X_train, y_train, sample_weight=sample_weight)
    best_params = grid.best_params_

    best_mlp = MLPClassifier(
        **best_params,
        random_state=42,
        max_iter=500,
        early_stopping=True,
        n_iter_no_change=10
    )
    best_mlp.fit(X_train, y_train, sample_weight=sample_weight)

    y_pred = best_mlp.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"Najlepsze hiperparametry: {best_params}")
    print(f"Accuracy dla wina {wine_type.upper()}: {acc*100:.2f}%\n")
    print("Raport Klasyfikacji:")
    print(classification_report(y_test, y_pred, zero_division=0))

    print("[2/2] Kalibracja modelu (Sigmoid & Isotonic)...")

    mlp_config = MLPClassifier(
        **best_params,
        random_state=42,
        max_iter=500,
        early_stopping=True,
        n_iter_no_change=10
    )

    calibrated_sig = CalibratedClassifierCV(estimator=mlp_config, method="sigmoid", cv=3)
    calibrated_sig.fit(X_train, y_train, sample_weight=sample_weight)
    y_prob_sig = calibrated_sig.predict_proba(X_test)

    calibrated_iso = CalibratedClassifierCV(estimator=mlp_config, method="isotonic", cv=3)
    calibrated_iso.fit(X_train, y_train, sample_weight=sample_weight)
    y_prob_iso = calibrated_iso.predict_proba(X_test)

    print(f"\n[Log loss]:")
    print(f"Log loss (Sigmoid):   {log_loss(y_test, y_prob_sig, labels=[0,1,2]):.4f}")
    print(f"Log loss (Isotonic):  {log_loss(y_test, y_prob_iso, labels=[0,1,2]):.4f}")
    
    # wybierz najlepszą kalibrację według log-loss i zapisz model(y)
    loss_sig = log_loss(y_test, y_prob_sig, labels=[0,1,2])
    loss_iso = log_loss(y_test, y_prob_iso, labels=[0,1,2])

    if loss_sig <= loss_iso:
        best_calib = "sigmoid"
        best_calibrated = calibrated_sig
    else:
        best_calib = "isotonic"
        best_calibrated = calibrated_iso

    os.makedirs("models", exist_ok=True)
    dump(best_calibrated, f"models/mlp_{wine_type}_{best_calib}.joblib")
    dump(best_mlp, f"models/mlp_{wine_type}_uncalibrated.joblib")

    print(f"Zapisano: models/mlp_{wine_type}_{best_calib}.joblib i models/mlp_{wine_type}_uncalibrated.joblib")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], "k:", label="Idealnie skalibrowany")

    y_test_bin = label_binarize(y_test, classes=[0,1,2])

    for i, name in enumerate(["slabe", "srednie", "dobre"]):
        plot_calibration(y_test_bin[:, i], y_prob_sig[:, i], f"MLP sigmoid {name}", ax)
        plot_calibration(y_test_bin[:, i], y_prob_iso[:, i], f"MLP isotonic {name}", ax)

    ax.set_ylabel("Frakcja w klasie pozytywnej")
    ax.set_xlabel("Srednia wartosc przewidywana")
    ax.legend(loc="lower right")
    ax.set_title(f"Krzywe kalibracji - Wino {wine_type.upper()}")
    plt.tight_layout()
    plt.savefig(f"results/mlp_calibration_{wine_type}.png")
    print(f"Zapisano wykres kalibracji do results/mlp_calibration_{wine_type}.png.")
    print("Liczba wykonanych epok:", best_mlp.n_iter_)
    
def main():
    if not os.path.exists("results"):
        os.makedirs("results")

    for wine in ["red", "white"]:
        process_and_evaluate(wine)


if __name__ == "__main__":
    main()