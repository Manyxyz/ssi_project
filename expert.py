import os
import numpy as np
import pandas as pd
from joblib import load
from sklearn.metrics import f1_score, log_loss

MODEL_MAP = {
    "red": {
        "rf": "models/rf_red_sigmoid.joblib",
        "svm": "models/svm_red_isotonic.joblib",
        "knn": "models/knn_red_isotonic.joblib",
        "mlp": "models/mlp_red_isotonic.joblib",
    },
    "white": {
        "rf": "models/rf_white_isotonic.joblib",
        "svm": "models/svm_white_isotonic.joblib",
        "knn": "models/knn_white_sigmoid.joblib",
        "mlp": "models/mlp_white_sigmoid.joblib",
    },
}

MODEL_WEIGHTS = {
    "rf": 0.35,
    "svm": 0.10,
    "knn": 0.25,
    "mlp": 0.30,
}

SCALER_MAP = {
    "red": "models/scaler_red.joblib",
    "white": "models/scaler_white.joblib",
}

CLASS_NAMES = {
    0: "slabe",
    1: "srednie",
    2: "dobre",
}

def map_quality(q):
    if q <= 4:
        return 0
    if q <= 6:
        return 1
    return 2

def load_feature_names(wine_type):
    X_train = pd.read_csv(f"X_train_{wine_type}_clean.csv", nrows=1)
    return [col for col in X_train.columns]

def load_models(wine_type):
    models = {}
    for name, path in MODEL_MAP[wine_type].items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Brak modelu: {path}")
        models[name] = load(path)
    return models

def load_scaler(wine_type):
    path = SCALER_MAP[wine_type]
    if os.path.exists(path):
        return load(path)
    return None

def ask_user_for_features(feature_names):
    print("\nPodaj wartości wszystkich cech w jednej linii, oddzielone średnikami (;)")
    print("Kolejność cech:")
    print("; ".join(feature_names))
    raw = input("\nWartości: ").strip().replace(",", ".")
    values = [float(x.strip()) for x in raw.split(";")]
    if len(values) != len(feature_names):
        raise ValueError(
            f"Podano {len(values)} wartości, a oczekiwano {len(feature_names)}."
        )
    return pd.DataFrame([values], columns=feature_names)

def prepare_user_input(wine_type, X_raw):
    scaler = load_scaler(wine_type)
    if scaler is None:
        print("Uwaga: brak scaler_*.joblib. Zakładam, że podałeś już wystandaryzowane cechy.")
        return X_raw
    arr = scaler.transform(X_raw)
    return pd.DataFrame(arr, columns=X_raw.columns)

def normalize_weights_for_models(weights, models):
    filtered = {k: weights[k] for k in models.keys() if k in weights}
    if not filtered:
        return {k: 1.0 / len(models) for k in models.keys()}
    s = sum(filtered.values())
    return {k: v / s for k, v in filtered.items()}

def safe_predict_proba(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    preds = model.predict(X)
    proba = np.zeros((len(preds), 3), dtype=float)
    for i, p in enumerate(preds):
        proba[i, int(p)] = 1.0
    return proba

def combine_soft_probabilities(models, X, weights):
    combined = np.zeros(3, dtype=float)
    probs_by_model = {}
    for name, model in models.items():
        proba = safe_predict_proba(model, X)[0]
        probs_by_model[name] = proba
        combined += weights[name] * proba
    combined = combined / combined.sum()
    return probs_by_model, combined

def threshold_decision(p, t0, t2):
    # Prefer class 0/2 if their calibrated probability crosses tuned threshold.
    if p[0] >= t0 and p[0] >= p[2]:
        return 0
    if p[2] >= t2 and p[2] > p[0]:
        return 2
    return 1

def compute_weights_from_logloss(models, wine_type):
    X_eval = pd.read_csv(f"X_test_{wine_type}_clean.csv")
    y_eval = pd.read_csv(f"y_test_{wine_type}_clean.csv")["quality"].apply(map_quality)

    inv = {}
    for name, model in models.items():
        proba = safe_predict_proba(model, X_eval)
        loss = log_loss(y_eval, proba, labels=[0, 1, 2])
        inv[name] = 1.0 / (loss + 1e-8)

    s = sum(inv.values())
    return {k: v / s for k, v in inv.items()}

def tune_thresholds(models, wine_type, weights):
    X_eval = pd.read_csv(f"X_test_{wine_type}_clean.csv")
    y_eval = pd.read_csv(f"y_test_{wine_type}_clean.csv")["quality"].apply(map_quality).to_numpy()

    combined_probs = []
    for i in range(len(X_eval)):
        x = X_eval.iloc[[i]]
        _, p = combine_soft_probabilities(models, x, weights)
        combined_probs.append(p)
    combined_probs = np.vstack(combined_probs)

    best = {"f1": -1.0, "t0": 0.20, "t2": 0.20}
    grid = np.arange(0.05, 0.51, 0.05)
    for t0 in grid:
        for t2 in grid:
            preds = np.array([threshold_decision(p, t0, t2) for p in combined_probs])
            score = f1_score(y_eval, preds, average="macro", zero_division=0)
            if score > best["f1"]:
                best = {"f1": score, "t0": float(t0), "t2": float(t2)}

    return best

def predict_hard(models, X):
    votes = []
    for model in models.values():
        pred = int(model.predict(X)[0])
        votes.append(pred)

    counts = np.bincount(votes, minlength=3)
    final_class = int(np.argmax(counts))
    return final_class, votes, counts

def predict_weighted(models, X, weights):
    weighted_counts = np.zeros(3, dtype=float)
    class_votes = {}

    for name, model in models.items():
        pred = int(model.predict(X)[0])
        class_votes[name] = pred
        weighted_counts[pred] += weights[name]

    final_class = int(np.argmax(weighted_counts))
    return final_class, class_votes, weighted_counts

def predict_soft(models, X, weights):
    probs_by_model, combined = combine_soft_probabilities(models, X, weights)
    final_class = int(np.argmax(combined))
    return final_class, probs_by_model, combined

def print_result(title, cls):
    print(f"{title}: {cls} ({CLASS_NAMES[cls]})")

def main():
    print("Dostępne typy wina: red, white")
    wine_type = input("Wybierz typ wina: ").strip().lower()

    if wine_type not in MODEL_MAP:
        raise ValueError("Wybierz tylko 'red' albo 'white'.")

    feature_names = load_feature_names(wine_type)
    models = load_models(wine_type)
    X_user_raw = ask_user_for_features(feature_names)
    X_user = prepare_user_input(wine_type, X_user_raw)

    use_auto_weights = input("Użyć automatycznych wag z log-loss? (y/N): ").strip().lower() == "y"
    if use_auto_weights:
        model_weights_local = compute_weights_from_logloss(models, wine_type)
        print("Wagi z log-loss:", model_weights_local)
    else:
        model_weights_local = normalize_weights_for_models(MODEL_WEIGHTS, models)

    tune_thr = input("Dostroić progi klasy 0/2 na X_test? (y/N): ").strip().lower() == "y"
    if tune_thr:
        best_thr = tune_thresholds(models, wine_type, model_weights_local)
        t0, t2 = best_thr["t0"], best_thr["t2"]
        print(f"Najlepsze progi: t0={t0:.2f}, t2={t2:.2f}, f1_macro={best_thr['f1']:.4f}")
    else:
        t0, t2 = 0.20, 0.20

    print("\n=== HARD VOTING ===")
    hard_cls, hard_votes, hard_counts = predict_hard(models, X_user)
    print("Głosy modeli:", hard_votes)
    print("Zliczenie klas:", hard_counts.tolist())
    print_result("Wynik hard voting", hard_cls)

    print("\n=== WEIGHTED VOTING ===")
    weighted_cls, weighted_votes, weighted_counts = predict_weighted(models, X_user, model_weights_local)
    print("Głosy modeli:", weighted_votes)
    print("Wagi:", model_weights_local)
    print("Ważone zliczenie klas:", weighted_counts.tolist())
    print_result("Wynik weighted voting", weighted_cls)

    print("\n=== SOFT VOTING ===")
    soft_cls, probs_by_model, combined = predict_soft(models, X_user, model_weights_local)
    for name, proba in probs_by_model.items():
        print(f"{name.upper()} -> {np.round(proba, 4).tolist()}")
    print("Połączone prawdopodobieństwa:", np.round(combined, 4).tolist())
    soft_thr_cls = threshold_decision(combined, t0=t0, t2=t2)
    print_result("Wynik soft voting (argmax)", soft_cls)
    print(f"Progi decyzyjne: t0={t0:.2f}, t2={t2:.2f}")
    print_result("Wynik soft voting (progowy)", soft_thr_cls)

if __name__ == "__main__":
    main()