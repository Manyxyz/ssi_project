import numpy as np
import pandas as pd

import skfuzzy as fuzz
from skfuzzy import control as ctrl

CLASS_NAMES = {0: "slabe", 1: "srednie", 2: "dobre"}


def load_feature_names():
    return [
        "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
        "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
        "pH", "sulphates", "alcohol"
    ]


def ask_user_for_features(feature_names):
    print("\nPodaj wartości wszystkich 11 cech w jednej linii, oddzielone średnikami (;)")
    print("Kolejność wprowadzania parametrów:")
    print("; ".join(feature_names))
    raw = input("\nWartości: ").strip().replace(",", ".")
    values = [float(x.strip()) for x in raw.split(";")]
    if len(values) != len(feature_names):
        raise ValueError(f"Podano {len(values)} wartości, a oczekiwano dokładnie {len(feature_names)}.")
    return pd.DataFrame([values], columns=feature_names)


def build_complete_fuzzy_system(wine_type):
    if wine_type == "white":
        fa  = ctrl.Antecedent(np.arange(3.8, 14.3, 0.1),      'fixed_acidity')
        va  = ctrl.Antecedent(np.arange(0.08, 1.15, 0.01),    'volatile_acidity')
        ca  = ctrl.Antecedent(np.arange(0.0, 1.7, 0.01),      'citric_acid')
        rs  = ctrl.Antecedent(np.arange(0.6, 66.0, 0.1),      'residual_sugar')
        ch  = ctrl.Antecedent(np.arange(0.009, 0.35, 0.001),  'chlorides')
        fsd = ctrl.Antecedent(np.arange(2.0, 290.0, 1.0),     'free_sulfur_dioxide')
        tsd = ctrl.Antecedent(np.arange(9.0, 441.0, 1.0),     'total_sulfur_dioxide')
        den = ctrl.Antecedent(np.arange(0.987, 1.039, 0.0001),'density')
        ph  = ctrl.Antecedent(np.arange(2.72, 3.82, 0.01),    'pH')
        su  = ctrl.Antecedent(np.arange(0.22, 1.1, 0.01),     'sulphates')
        alc = ctrl.Antecedent(np.arange(8.0, 14.3, 0.1),      'alcohol')
    else:
        fa  = ctrl.Antecedent(np.arange(4.6, 16.0, 0.1),      'fixed_acidity')
        va  = ctrl.Antecedent(np.arange(0.12, 1.58, 0.01),     'volatile_acidity')
        ca  = ctrl.Antecedent(np.arange(0.0, 1.0, 0.01),       'citric_acid')
        rs  = ctrl.Antecedent(np.arange(0.9, 15.6, 0.1),       'residual_sugar')
        ch  = ctrl.Antecedent(np.arange(0.012, 0.615, 0.001),  'chlorides')
        fsd = ctrl.Antecedent(np.arange(1.0, 73.0, 1.0),       'free_sulfur_dioxide')
        tsd = ctrl.Antecedent(np.arange(6.0, 290.0, 1.0),      'total_sulfur_dioxide')
        den = ctrl.Antecedent(np.arange(0.990, 1.004, 0.0001), 'density')
        ph  = ctrl.Antecedent(np.arange(2.74, 4.05, 0.01),     'pH')
        su  = ctrl.Antecedent(np.arange(0.33, 2.0, 0.01),      'sulphates')
        alc = ctrl.Antecedent(np.arange(8.4, 14.9, 0.1),       'alcohol')

    quality = ctrl.Consequent(np.arange(0, 2.01, 0.01), 'quality')

   
    if wine_type == "white":
        alc['niski']  = fuzz.trapmf(alc.universe, [8.0, 8.0, 9.0, 9.6])
        alc['sredni'] = fuzz.trimf(alc.universe,  [9.2, 10.2, 11.2])
        alc['wysoki'] = fuzz.trapmf(alc.universe, [10.4, 11.0, 14.3, 14.3])

        va['niska']  = fuzz.trapmf(va.universe, [0.08, 0.08, 0.20, 0.26])
        va['srednia']= fuzz.trimf(va.universe,  [0.23, 0.28, 0.35])
        va['wysoka'] = fuzz.trapmf(va.universe, [0.32, 0.40, 1.15, 1.15])

        den['niski']  = fuzz.trapmf(den.universe, [0.987, 0.987, 0.9905, 0.9930])

        den['sredni'] = fuzz.trimf(den.universe,  [0.9918, 0.9942, 0.9962])
        den['wysoki'] = fuzz.trapmf(den.universe, [0.9952, 0.9972, 1.039, 1.039])

        ch['niski']  = fuzz.trapmf(ch.universe, [0.009, 0.009, 0.030, 0.037])
        ch['sredni'] = fuzz.trimf(ch.universe,  [0.034, 0.043, 0.053])
        ch['wysoki'] = fuzz.trapmf(ch.universe, [0.049, 0.062, 0.346, 0.346])


        fsd['niski']  = fuzz.trapmf(fsd.universe, [2, 2, 8, 14])
        fsd['sredni'] = fuzz.trimf(fsd.universe,  [12, 32, 48])
        fsd['wysoki'] = fuzz.trapmf(fsd.universe, [42, 56, 290, 290])

        fa['niska']  = fuzz.trapmf(fa.universe, [3.8, 3.8, 6.1, 6.6])
        fa['srednia']= fuzz.trimf(fa.universe,  [6.3, 6.8, 7.4])
        fa['wysoka'] = fuzz.trapmf(fa.universe, [7.1, 8.0, 14.2, 14.2])

        tsd['niski']  = fuzz.trapmf(tsd.universe, [9, 9, 95, 115])
        tsd['sredni'] = fuzz.trimf(tsd.universe,  [105, 135, 165])
        tsd['wysoki'] = fuzz.trapmf(tsd.universe, [155, 180, 441, 441])

        su['niski']  = fuzz.trapmf(su.universe, [0.22, 0.22, 0.38, 0.43])
        su['sredni'] = fuzz.trimf(su.universe,  [0.40, 0.47, 0.55])
        su['wysoki'] = fuzz.trapmf(su.universe, [0.52, 0.62, 1.10, 1.10])

        ph['niski']  = fuzz.trapmf(ph.universe, [2.72, 2.72, 3.07, 3.14])
        ph['sredni'] = fuzz.trimf(ph.universe,  [3.10, 3.18, 3.27])
        ph['wysoki'] = fuzz.trapmf(ph.universe, [3.24, 3.32, 3.82, 3.82])

        rs['malo']    = fuzz.trapmf(rs.universe, [0.6, 0.6, 1.7, 3.5])
        rs['srednio'] = fuzz.trimf(rs.universe,  [2.5, 5.5, 10.0])
        rs['duzo']    = fuzz.trapmf(rs.universe, [8.5, 13.0, 66.0, 66.0])

        ca['malo']    = fuzz.trapmf(ca.universe, [0.0, 0.0, 0.25, 0.30])
        ca['srednio'] = fuzz.trimf(ca.universe,  [0.28, 0.32, 0.40])
        ca['duzo']    = fuzz.trapmf(ca.universe, [0.38, 0.48, 1.66, 1.66])

    else:
        alc['niski']  = fuzz.trapmf(alc.universe, [8.4, 8.4, 9.4, 9.9])
        alc['sredni'] = fuzz.trimf(alc.universe,  [9.5, 10.3, 11.2])
        alc['wysoki'] = fuzz.trapmf(alc.universe, [10.8, 11.6, 14.9, 14.9])

        va['niski']  = fuzz.trapmf(va.universe, [0.12, 0.12, 0.32, 0.42])
        va['sredni'] = fuzz.trimf(va.universe,  [0.38, 0.52, 0.66])
        va['wysoki'] = fuzz.trapmf(va.universe, [0.60, 0.74, 1.58, 1.58])

        su['niski']  = fuzz.trapmf(su.universe, [0.33, 0.33, 0.51, 0.58])
        su['sredni'] = fuzz.trimf(su.universe,  [0.54, 0.64, 0.75])
        su['wysoki'] = fuzz.trapmf(su.universe, [0.70, 0.82, 2.0, 2.0])

        ca['niski']  = fuzz.trapmf(ca.universe, [0.0, 0.0, 0.09, 0.16])
        ca['sredni'] = fuzz.trimf(ca.universe,  [0.12, 0.26, 0.42])
        ca['wysoki'] = fuzz.trapmf(ca.universe, [0.38, 0.48, 1.0, 1.0])

        ph['niski']  = fuzz.trapmf(ph.universe, [2.74, 2.74, 3.18, 3.26])
        ph['sredni'] = fuzz.trimf(ph.universe,  [3.22, 3.31, 3.42])
        ph['wysoki'] = fuzz.trapmf(ph.universe, [3.38, 3.48, 4.05, 4.05])

        fa['niski']  = fuzz.trapmf(fa.universe, [4.6, 4.6, 6.8, 7.4])
        fa['sredni'] = fuzz.trimf(fa.universe,  [7.0, 7.9, 9.3])
        fa['wysoki'] = fuzz.trapmf(fa.universe, [8.8, 10.2, 16.0, 16.0])

        tsd['niski']  = fuzz.trapmf(tsd.universe, [6, 6, 22, 34])
        tsd['sredni'] = fuzz.trimf(tsd.universe,  [28, 44, 65])
        tsd['wysoki'] = fuzz.trapmf(tsd.universe, [58, 78, 290, 290])

        ch['niski']  = fuzz.trapmf(ch.universe, [0.012, 0.012, 0.065, 0.075])
        ch['sredni'] = fuzz.trimf(ch.universe,  [0.070, 0.080, 0.093])
        ch['wysoki'] = fuzz.trapmf(ch.universe, [0.088, 0.105, 0.615, 0.615])

        den['niski']  = fuzz.trapmf(den.universe, [0.990, 0.990, 0.9945, 0.9960])
        den['sredni'] = fuzz.trimf(den.universe,  [0.9952, 0.9967, 0.9982])
        den['wysoki'] = fuzz.trapmf(den.universe, [0.9975, 0.9990, 1.004, 1.004])

        rs['niski']  = fuzz.trapmf(rs.universe, [0.9, 0.9, 1.8, 2.1])
        rs['sredni'] = fuzz.trimf(rs.universe,  [1.9, 2.2, 3.0])
        rs['wysoki'] = fuzz.trapmf(rs.universe, [2.6, 3.5, 15.6, 15.6])

        fsd['niski']  = fuzz.trapmf(fsd.universe, [1, 1, 6, 10])
        fsd['sredni'] = fuzz.trimf(fsd.universe,  [8, 14, 22])
        fsd['wysoki'] = fuzz.trapmf(fsd.universe, [19, 28, 73, 73])

    quality['slabe']   = fuzz.trimf(quality.universe, [0.0, 0.0, 0.85])
    quality['srednie'] = fuzz.trimf(quality.universe, [0.45, 1.0, 1.55])
    quality['dobre']   = fuzz.trimf(quality.universe, [1.15, 2.0, 2.0])

    if wine_type == "white":
        rules = [
            ctrl.Rule(alc['wysoki'] & den['niski'], quality['dobre']),

            ctrl.Rule(alc['wysoki'] & ch['niski'], quality['dobre']),

            ctrl.Rule(alc['wysoki'] & den['niski'] & va['niska'], quality['dobre']),

            ctrl.Rule(den['niski'] & va['niska'] & ch['niski'], quality['dobre']),

            ctrl.Rule(alc['wysoki'] & tsd['niski'], quality['dobre']),

            ctrl.Rule(va['wysoka'] & alc['niski'], quality['slabe']),

            ctrl.Rule(va['wysoka'] & den['wysoki'], quality['slabe']),

            ctrl.Rule(fsd['niski'] & alc['niski'], quality['slabe']),

            ctrl.Rule(fsd['niski'] & ch['wysoki'], quality['slabe']),

            ctrl.Rule(den['wysoki'] & ch['wysoki'], quality['slabe']),


            ctrl.Rule(alc['sredni'] & den['sredni'], quality['srednie']),

            ctrl.Rule(alc['niski'] & den['sredni'], quality['srednie']),

            ctrl.Rule(alc['niski'] & den['wysoki'], quality['srednie']),

            ctrl.Rule(alc['sredni'] & va['srednia'], quality['srednie']),

            ctrl.Rule(alc['wysoki'] & den['wysoki'], quality['srednie']),

            ctrl.Rule(alc['sredni'] & tsd['wysoki'], quality['srednie']),


            ctrl.Rule(alc['niski'] & va['niska'] & ch['niski'], quality['srednie']),
        ]


    else:
        rules = [
            ctrl.Rule(alc['wysoki'] & va['niski'], quality['dobre']),
            ctrl.Rule(alc['wysoki'] & su['wysoki'], quality['dobre']),
            ctrl.Rule(va['niski'] & su['wysoki'] & ca['wysoki'], quality['dobre']),
            ctrl.Rule(alc['wysoki'] & ca['wysoki'] & ch['niski'], quality['dobre']),

            ctrl.Rule(va['wysoki'] & alc['niski'], quality['slabe']),
            ctrl.Rule(va['wysoki'] & ca['niski'], quality['slabe']),
            ctrl.Rule(va['wysoki'] & su['niski'], quality['slabe']),
            ctrl.Rule(va['wysoki'] & ch['wysoki'], quality['slabe']),
            ctrl.Rule(ch['wysoki'] & ca['niski'] & su['niski'], quality['slabe']),

            ctrl.Rule(alc['wysoki'] & va['sredni'], quality['srednie']),
            ctrl.Rule(alc['sredni'] & va['niski'], quality['srednie']),
            ctrl.Rule(alc['sredni'] & va['sredni'], quality['srednie']),
            ctrl.Rule(su['sredni'] & ca['sredni'], quality['srednie']),
            ctrl.Rule(alc['niski'] & va['niski'] & su['wysoki'], quality['srednie']),
        ]

    fuzzy_control = ctrl.ControlSystem(rules)
    return ctrl.ControlSystemSimulation(fuzzy_control)


def main():
    print("=== SYSTEM ROZMYTY MAMDANIEGO – OCENA JAKOŚCI WINA ===")
    print("(v2: poprawione MF i reguły dla białego wina)")

    wine_type = input("\nWybierz typ wina (red/white): ").strip().lower()
    if wine_type not in ["red", "white"]:
        raise ValueError("Dozwolone opcje to wyłącznie 'red' lub 'white'.")

    feature_names = load_feature_names()
    X_user_raw = ask_user_for_features(feature_names)

    fuzzy_engine = build_complete_fuzzy_system(wine_type)

    for feature in feature_names:
        fuzzy_name = feature.replace(" ", "_")
        try:
            fuzzy_engine.input[fuzzy_name] = X_user_raw[feature].values[0]
        except (ValueError, KeyError):
            pass

    try:
        fuzzy_engine.compute()
        centroid = fuzzy_engine.output['quality']

        if centroid < 0.80:
            final_class = 0
        elif centroid > 1.20:
            final_class = 2
        else:
            final_class = 1

        print("\n" + "=" * 50)
        print("  WYNIK SYSTEMU EKSPERCKIEGO")
        print("=" * 50)
        print(f"  Wartość wyjściowa (centroid): {centroid:.4f}")
        print(f"  Klasyfikacja:  Klasa {final_class} ({CLASS_NAMES[final_class].upper()})")
        print("=" * 50)

    except Exception as e:
        print(f"\n[Błąd obliczeń]: {e}")
        print(f"Werdykt bezpieczny (Fallback): Klasa 1 ({CLASS_NAMES[1].upper()})")


if __name__ == "__main__":
    main()