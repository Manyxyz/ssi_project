import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, log_loss
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.preprocessing import label_binarize
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

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
    counts = y_train.value_counts()
    majority = counts.idxmax()
    min_count = counts.min()
    k_neighbors = 1 if min_count <= 2 else 2

    target = {}
    for cls, cnt in counts.items():
        if cls != majority:
            target[cls] = min(int(cnt * 1.2), int(counts[majority]))

    smote_train = SMOTE(
        random_state=42,
        k_neighbors=k_neighbors,
        sampling_strategy=target
    )
    smote_calib = SMOTE(
        random_state=42,
        k_neighbors=k_neighbors,
        sampling_strategy=target
    )
    print("[1/2] Trening KNN i GridSearch...")
    base_knn = KNeighborsClassifier()

    pipe = Pipeline([
        ("smote", smote_train),
        ("knn", base_knn)
    ])

    param_grid = {
        "knn__n_neighbors": [3, 5, 7, 9, 11, 15, 21],
        "knn__weights": ["uniform", "distance"],
        "knn__metric": ["minkowski"],
        "knn__p": [1, 2]
    }

    grid = GridSearchCV(
        pipe,
        param_grid,
        cv=3,
        n_jobs=-1,
        scoring="f1_macro"
    )
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_

    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"Najlepsze hiperparametry: {grid.best_params_}")
    print(f"Accuracy dla wina {wine_type.upper()}: {acc*100:.2f}%\n")
    print("Raport Klasyfikacji:")
    print(classification_report(y_test, y_pred, zero_division=0))

    print("[2/2] Kalibracja modelu (Sigmoid & Isotonic)...")

    calib_estimator = Pipeline([
        ("smote", smote_calib),
        ("knn", KNeighborsClassifier())
    ])
    calib_estimator.set_params(**grid.best_params_)

    calibrated_sig = CalibratedClassifierCV(estimator=calib_estimator, method="sigmoid", cv=3)
    calibrated_sig.fit(X_train, y_train)
    y_prob_sig = calibrated_sig.predict_proba(X_test)

    calibrated_iso = CalibratedClassifierCV(estimator=calib_estimator, method="isotonic", cv=3)
    calibrated_iso.fit(X_train, y_train)
    y_prob_iso = calibrated_iso.predict_proba(X_test)

    print(f"\n[Log loss]:")
    print(f"Log loss (Sigmoid):   {log_loss(y_test, y_prob_sig, labels=[0,1,2]):.4f}")
    print(f"Log loss (Isotonic):  {log_loss(y_test, y_prob_iso, labels=[0,1,2]):.4f}")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], "k:", label="Idealnie skalibrowany")

    y_test_bin = label_binarize(y_test, classes=[0,1,2])

    for i, name in enumerate(["slabe", "srednie", "dobre"]):
        plot_calibration(y_test_bin[:, i], y_prob_sig[:, i], f"KNN sigmoid {name}", ax)
        plot_calibration(y_test_bin[:, i], y_prob_iso[:, i], f"KNN isotonic {name}", ax)

    ax.set_ylabel("Frakcja w klasie pozytywnej")
    ax.set_xlabel("Srednia wartosc przewidywana")
    ax.legend(loc="lower right")
    ax.set_title(f"Krzywe kalibracji - Wino {wine_type.upper()}")
    plt.tight_layout()
    plt.savefig(f"results/knn_calibration_{wine_type}.png")
    print(f"Zapisano wykres kalibracji do `results/knn_calibration_{wine_type}.png`.")

def main():
    if not os.path.exists("results"):
        os.makedirs("results")

    for wine in ["red", "white"]:
        process_and_evaluate(wine)

if __name__ == "__main__":
    main()