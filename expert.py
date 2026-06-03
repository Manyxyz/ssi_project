import os
import numpy as np
import pandas as pd
from joblib import load

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

def ask_user_for_features(feature_names):
    values = []
    print("\nPodaj wartości cech w kolejności poniżej:")
    for feature in feature_names:
        raw = input(f"{feature}: ").strip().replace(",", ".")
        values.append(float(raw))
    return pd.DataFrame([values], columns=feature_names)

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
    combined = np.zeros(3, dtype=float)
    probs_by_model = {}

    for name, model in models.items():
        proba = model.predict_proba(X)[0]
        probs_by_model[name] = proba
        combined += weights[name] * proba

    combined = combined / combined.sum()
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
    X_user = ask_user_for_features(feature_names)

    print("\n=== HARD VOTING ===")
    hard_cls, hard_votes, hard_counts = predict_hard(models, X_user)
    print("Głosy modeli:", hard_votes)
    print("Zliczenie klas:", hard_counts.tolist())
    print_result("Wynik hard voting", hard_cls)

    print("\n=== WEIGHTED VOTING ===")
    weighted_cls, weighted_votes, weighted_counts = predict_weighted(models, X_user, MODEL_WEIGHTS)
    print("Głosy modeli:", weighted_votes)
    print("Wagi:", MODEL_WEIGHTS)
    print("Ważone zliczenie klas:", weighted_counts.tolist())
    print_result("Wynik weighted voting", weighted_cls)

    print("\n=== SOFT VOTING ===")
    soft_cls, probs_by_model, combined = predict_soft(models, X_user, MODEL_WEIGHTS)
    for name, proba in probs_by_model.items():
        print(f"{name.upper()} -> {np.round(proba, 4).tolist()}")
    print("Połączone prawdopodobieństwa:", np.round(combined, 4).tolist())
    print_result("Wynik soft voting", soft_cls)

if __name__ == "__main__":
    main()