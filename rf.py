import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, log_loss
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.preprocessing import label_binarize

def load_wine_data(wine_type):
    # Wczytywanie po nazwie typu ('red' lub 'white')
    X_train = pd.read_csv(f'X_train_{wine_type}_clean.csv')
    X_test = pd.read_csv(f'X_test_{wine_type}_clean.csv')
    y_train = pd.read_csv(f'y_train_{wine_type}_clean.csv')
    y_test = pd.read_csv(f'y_test_{wine_type}_clean.csv')

    y_train_3 = y_train['quality'].apply(map_quality)
    y_test_3 = y_test['quality'].apply(map_quality)

    return X_train, X_test, y_train_3, y_test_3

def plot_calibration(y_true, proba, name, ax):
    fraction_of_positives, mean_predicted_value = calibration_curve(y_true, proba, n_bins=10)
    ax.plot(mean_predicted_value, fraction_of_positives, "s-", label=f"{name}")

def map_quality(q):
    if q <= 4:
        return 0  # slabe (3-4)
    if q <= 6:
        return 1  # srednie (5-6)
    return 2      # dobre (7-9)

def process_and_evaluate(wine_type):
    print(f"\n{'='*50}")
    print(f"--- ROZPOCZYNAM ANALIZĘ DLA WINA: {wine_type.upper()} ---")
    print(f"{'='*50}")

    X_train, X_test, y_train, y_test = load_wine_data(wine_type)
    print(f"Dane treningowe: {X_train.shape}, Dane testowe: {X_test.shape}\n")

    print("[1/2] Trening RandomForest i GridSearch...")
    base_rf = RandomForestClassifier(
        random_state=42,
        class_weight="balanced"
    )
    param_grid = {
        "n_estimators": [200, 400, 600],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2, 4],
        "class_weight": ["balanced", "balanced_subsample"]
    }
    
    grid = GridSearchCV(base_rf, param_grid, cv=3, n_jobs=-1, scoring="f1_macro")
    grid.fit(X_train, y_train)
    best_rf = grid.best_estimator_
    
    y_pred = best_rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"Najlepsze hiperparametry: {grid.best_params_}")
    print(f"Accuracy dla wina {wine_type.upper()}: {acc*100:.2f}%\n")
    print("Raport Klasyfikacji:")
    print(classification_report(y_test, y_pred))

    print("[2/2] Kalibracja modelu (Sigmoid & Isotonic) [Opcja A: Czysta Konfiguracja]...")
    
    # OPcja A: Wyciągamy samą konfigurację modelu do kalibratora by uniknąć wycieku (Data Leakage)
    if hasattr(best_rf, 'get_params'):
         rf_config = RandomForestClassifier(**grid.best_params_, random_state=42)
    else:    
         rf_config = RandomForestClassifier(random_state=42)

    # Sigmoid na czystym configu, uruchamiany przez cv=3
    calibrated_sig = CalibratedClassifierCV(estimator=rf_config, method='sigmoid', cv=3)
    calibrated_sig.fit(X_train, y_train)
    y_prob_sig = calibrated_sig.predict_proba(X_test)
    
    # Isotonic na czystym configu, uruchamiany przez cv=3
    calibrated_iso = CalibratedClassifierCV(estimator=rf_config, method='isotonic', cv=3)
    calibrated_iso.fit(X_train, y_train)
    y_prob_iso = calibrated_iso.predict_proba(X_test)
    
    # Bazowe prawdopodobieństwo liczymy na pełnym wyuczonym best_rf
    y_prob_base = best_rf.predict_proba(X_test)  # shape (n, 3)

    print(f"\n[Log loss (Bazowy RF)]:")
    print(f"Log loss (Bazowy RF): {log_loss(y_test, y_prob_base, labels=[0,1,2]):.4f}")
    print(f"Log loss (Sigmoid):   {log_loss(y_test, y_prob_sig, labels=[0,1,2]):.4f}")
    print(f"Log loss (Isotonic):  {log_loss(y_test, y_prob_iso, labels=[0,1,2]):.4f}")

    # Tworzenie wykresu
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], "k:", label="Idealnie skalibrowany")
    
    y_test_bin = label_binarize(y_test, classes=[0,1,2])

    for i, name in enumerate(["slabe", "srednie", "dobre"]):
        plot_calibration(y_test_bin[:, i], y_prob_base[:, i], f"RF {name}", ax)
    
    ax.set_ylabel("Frakcja w klasie pozytywnej")
    ax.set_xlabel("Średnia wartość przewidywana")
    ax.legend(loc="lower right")
    ax.set_title(f"Krzywe kalibracji - Wino {wine_type.upper()}")
    plt.tight_layout()
    plt.savefig(f'results/rf_calibration_{wine_type}.png')
    print(f"Zapisano wykres kalibracji do `results/rf_calibration_{wine_type}.png`.")

def main():
    if not os.path.exists('results'):
        os.makedirs('results')
    
    # Niezależne wywołanie dla czerwonego i białego wina
    for wine in ['red', 'white']:
        process_and_evaluate(wine)

if __name__ == '__main__':
    main()