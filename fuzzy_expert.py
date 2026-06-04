import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score

CLASS_NAMES  = {0: "slabe", 1: "srednie", 2: "dobre"}
CLASS_COLORS = {"slabe": "#e74c3c", "srednie": "#f39c12", "dobre": "#27ae60"}
WINE_LABEL   = {"white": "białe", "red": "czerwone"}

def build_complete_fuzzy_system(wine_type):
    if wine_type == "white":
        fa  = ctrl.Antecedent(np.arange(3.8,  14.3,  0.1),      'fixed_acidity')
        va  = ctrl.Antecedent(np.arange(0.08,  1.15,  0.01),    'volatile_acidity')
        ca  = ctrl.Antecedent(np.arange(0.0,   1.7,   0.01),    'citric_acid')
        rs  = ctrl.Antecedent(np.arange(0.6,  66.0,   0.1),     'residual_sugar')
        ch  = ctrl.Antecedent(np.arange(0.009, 0.35,  0.001),   'chlorides')
        fsd = ctrl.Antecedent(np.arange(2.0,  290.0,  1.0),     'free_sulfur_dioxide')
        tsd = ctrl.Antecedent(np.arange(9.0,  441.0,  1.0),     'total_sulfur_dioxide')
        den = ctrl.Antecedent(np.arange(0.987, 1.039, 0.0001),  'density')
        ph  = ctrl.Antecedent(np.arange(2.72,  3.82,  0.01),    'pH')
        su  = ctrl.Antecedent(np.arange(0.22,  1.1,   0.01),    'sulphates')
        alc = ctrl.Antecedent(np.arange(8.0,  14.3,   0.1),     'alcohol')

        alc['niski']  = fuzz.trapmf(alc.universe, [8.0, 8.0,  9.0,  9.6])
        alc['sredni'] = fuzz.trimf (alc.universe, [9.2, 10.2, 11.2])
        alc['wysoki'] = fuzz.trapmf(alc.universe, [10.4, 11.0, 14.3, 14.3])

        va['niska']   = fuzz.trapmf(va.universe, [0.08, 0.08, 0.20, 0.26])
        va['srednia'] = fuzz.trimf (va.universe, [0.23, 0.28, 0.35])
        va['wysoka']  = fuzz.trapmf(va.universe, [0.32, 0.40, 1.15, 1.15])

        den['niski']  = fuzz.trapmf(den.universe, [0.987,  0.987,  0.9905, 0.9930])
        den['sredni'] = fuzz.trimf (den.universe, [0.9918, 0.9942, 0.9962])
        den['wysoki'] = fuzz.trapmf(den.universe, [0.9952, 0.9972, 1.039,  1.039])

        ch['niski']   = fuzz.trapmf(ch.universe, [0.009, 0.009, 0.030, 0.037])
        ch['sredni']  = fuzz.trimf (ch.universe, [0.034, 0.043, 0.053])
        ch['wysoki']  = fuzz.trapmf(ch.universe, [0.049, 0.062, 0.346, 0.346])

        fsd['niski']  = fuzz.trapmf(fsd.universe, [2,  2,   8,  14])
        fsd['sredni'] = fuzz.trimf (fsd.universe, [12, 32,  48])
        fsd['wysoki'] = fuzz.trapmf(fsd.universe, [42, 56, 290, 290])

        tsd['niski']  = fuzz.trapmf(tsd.universe, [9,   9,   95, 115])
        tsd['sredni'] = fuzz.trimf (tsd.universe, [105, 135, 165])
        tsd['wysoki'] = fuzz.trapmf(tsd.universe, [155, 180, 441, 441])

        fa['niska']   = fuzz.trapmf(fa.universe, [3.8, 3.8, 6.1, 6.6])
        fa['srednia'] = fuzz.trimf (fa.universe, [6.3, 6.8, 7.4])
        fa['wysoka']  = fuzz.trapmf(fa.universe, [7.1, 8.0, 14.2, 14.2])

        su['niski']   = fuzz.trapmf(su.universe, [0.22, 0.22, 0.38, 0.43])
        su['sredni']  = fuzz.trimf (su.universe, [0.40, 0.47, 0.55])
        su['wysoki']  = fuzz.trapmf(su.universe, [0.52, 0.62, 1.10, 1.10])

        ph['niski']   = fuzz.trapmf(ph.universe, [2.72, 2.72, 3.07, 3.14])
        ph['sredni']  = fuzz.trimf (ph.universe, [3.10, 3.18, 3.27])
        ph['wysoki']  = fuzz.trapmf(ph.universe, [3.24, 3.32, 3.82, 3.82])

        rs['malo']    = fuzz.trapmf(rs.universe, [0.6, 0.6,  1.7,  3.5])
        rs['srednio'] = fuzz.trimf (rs.universe, [2.5, 5.5, 10.0])
        rs['duzo']    = fuzz.trapmf(rs.universe, [8.5, 13.0, 66.0, 66.0])

        ca['malo']    = fuzz.trapmf(ca.universe, [0.0,  0.0,  0.25, 0.30])
        ca['srednio'] = fuzz.trimf (ca.universe, [0.28, 0.32, 0.40])
        ca['duzo']    = fuzz.trapmf(ca.universe, [0.38, 0.48, 1.66, 1.66])

    else:  # red
        fa  = ctrl.Antecedent(np.arange(4.6,  16.0,  0.1),     'fixed_acidity')
        va  = ctrl.Antecedent(np.arange(0.12,  1.58,  0.01),   'volatile_acidity')
        ca  = ctrl.Antecedent(np.arange(0.0,   1.0,   0.01),   'citric_acid')
        rs  = ctrl.Antecedent(np.arange(0.9,  15.6,   0.1),    'residual_sugar')
        ch  = ctrl.Antecedent(np.arange(0.012, 0.615, 0.001),  'chlorides')
        fsd = ctrl.Antecedent(np.arange(1.0,  73.0,   1.0),    'free_sulfur_dioxide')
        tsd = ctrl.Antecedent(np.arange(6.0,  290.0,  1.0),    'total_sulfur_dioxide')
        den = ctrl.Antecedent(np.arange(0.990, 1.004, 0.0001), 'density')
        ph  = ctrl.Antecedent(np.arange(2.74,  4.05,  0.01),   'pH')
        su  = ctrl.Antecedent(np.arange(0.33,  2.0,   0.01),   'sulphates')
        alc = ctrl.Antecedent(np.arange(8.4,  14.9,   0.1),    'alcohol')

        alc['niski']  = fuzz.trapmf(alc.universe, [8.4,  8.4,  9.4,  9.9])
        alc['sredni'] = fuzz.trimf (alc.universe, [9.5,  10.3, 11.2])
        alc['wysoki'] = fuzz.trapmf(alc.universe, [10.8, 11.6, 14.9, 14.9])

        va['niski']   = fuzz.trapmf(va.universe, [0.12, 0.12, 0.32, 0.42])
        va['sredni']  = fuzz.trimf (va.universe, [0.38, 0.52, 0.66])
        va['wysoki']  = fuzz.trapmf(va.universe, [0.60, 0.74, 1.58, 1.58])

        su['niski']   = fuzz.trapmf(su.universe, [0.33, 0.33, 0.51, 0.58])
        su['sredni']  = fuzz.trimf (su.universe, [0.54, 0.64, 0.75])
        su['wysoki']  = fuzz.trapmf(su.universe, [0.70, 0.82, 2.0,  2.0])

        ca['niski']   = fuzz.trapmf(ca.universe, [0.0,  0.0,  0.09, 0.16])
        ca['sredni']  = fuzz.trimf (ca.universe, [0.12, 0.26, 0.42])
        ca['wysoki']  = fuzz.trapmf(ca.universe, [0.38, 0.48, 1.0,  1.0])

        ph['niski']   = fuzz.trapmf(ph.universe, [2.74, 2.74, 3.18, 3.26])
        ph['sredni']  = fuzz.trimf (ph.universe, [3.22, 3.31, 3.42])
        ph['wysoki']  = fuzz.trapmf(ph.universe, [3.38, 3.48, 4.05, 4.05])

        fa['niski']   = fuzz.trapmf(fa.universe, [4.6,  4.6,  6.8,  7.4])
        fa['sredni']  = fuzz.trimf (fa.universe, [7.0,  7.9,  9.3])
        fa['wysoki']  = fuzz.trapmf(fa.universe, [8.8,  10.2, 16.0, 16.0])

        tsd['niski']  = fuzz.trapmf(tsd.universe, [6,  6,   22,  34])
        tsd['sredni'] = fuzz.trimf (tsd.universe, [28, 44,  65])
        tsd['wysoki'] = fuzz.trapmf(tsd.universe, [58, 78, 290, 290])

        ch['niski']   = fuzz.trapmf(ch.universe, [0.012, 0.012, 0.065, 0.075])
        ch['sredni']  = fuzz.trimf (ch.universe, [0.070, 0.080, 0.093])
        ch['wysoki']  = fuzz.trapmf(ch.universe, [0.088, 0.105, 0.615, 0.615])

        den['niski']  = fuzz.trapmf(den.universe, [0.990,  0.990,  0.9945, 0.9960])
        den['sredni'] = fuzz.trimf (den.universe, [0.9952, 0.9967, 0.9982])
        den['wysoki'] = fuzz.trapmf(den.universe, [0.9975, 0.9990, 1.004,  1.004])

        rs['niski']   = fuzz.trapmf(rs.universe, [0.9, 0.9, 1.8, 2.1])
        rs['sredni']  = fuzz.trimf (rs.universe, [1.9, 2.2, 3.0])
        rs['wysoki']  = fuzz.trapmf(rs.universe, [2.6, 3.5, 15.6, 15.6])

        fsd['niski']  = fuzz.trapmf(fsd.universe, [1,  1,   6,  10])
        fsd['sredni'] = fuzz.trimf (fsd.universe, [8,  14,  22])
        fsd['wysoki'] = fuzz.trapmf(fsd.universe, [19, 28,  73,  73])

    quality = ctrl.Consequent(
        np.arange(0, 2.01, 0.01), 'quality',
        defuzzify_method='centroid'
    )
    quality['slabe']   = fuzz.trimf(quality.universe, [0.0,  0.0,  0.85])
    quality['srednie'] = fuzz.trimf(quality.universe, [0.45, 1.0,  1.55])
    quality['dobre']   = fuzz.trimf(quality.universe, [1.15, 2.0,  2.0])

    if wine_type == "white":
        rules = [
            ctrl.Rule(alc['wysoki'] & den['niski'],               quality['dobre']),
            ctrl.Rule(alc['wysoki'] & ch['niski'],                quality['dobre']),
            ctrl.Rule(alc['wysoki'] & den['niski'] & va['niska'], quality['dobre']),
            ctrl.Rule(den['niski']  & va['niska']  & ch['niski'], quality['dobre']),
            ctrl.Rule(alc['wysoki'] & tsd['niski'],               quality['dobre']),
            ctrl.Rule(va['wysoka']  & alc['niski'],               quality['slabe']),
            ctrl.Rule(va['wysoka']  & den['wysoki'],              quality['slabe']),
            ctrl.Rule(fsd['niski']  & alc['niski'],               quality['slabe']),
            ctrl.Rule(fsd['niski']  & ch['wysoki'],               quality['slabe']),
            ctrl.Rule(den['wysoki'] & ch['wysoki'],               quality['slabe']),
            ctrl.Rule(alc['sredni'] & den['sredni'],              quality['srednie']),
            ctrl.Rule(alc['niski']  & den['sredni'],              quality['srednie']),
            ctrl.Rule(alc['niski']  & den['wysoki'],              quality['srednie']),
            ctrl.Rule(alc['sredni'] & va['srednia'],              quality['srednie']),
            ctrl.Rule(alc['wysoki'] & den['wysoki'],              quality['srednie']),
            ctrl.Rule(alc['sredni'] & tsd['wysoki'],              quality['srednie']),
            ctrl.Rule(alc['niski']  & va['niska'] & ch['niski'],  quality['srednie']),
        ]
    else:
        rules = [
            ctrl.Rule(alc['wysoki'] & va['niski'],                quality['dobre']),
            ctrl.Rule(alc['wysoki'] & su['wysoki'],               quality['dobre']),
            ctrl.Rule(va['niski']   & su['wysoki'] & ca['wysoki'],quality['dobre']),
            ctrl.Rule(alc['wysoki'] & ca['wysoki'] & ch['niski'], quality['dobre']),
            ctrl.Rule(va['wysoki']  & alc['niski'],               quality['slabe']),
            ctrl.Rule(va['wysoki']  & ca['niski'],                quality['slabe']),
            ctrl.Rule(va['wysoki']  & su['niski'],                quality['slabe']),
            ctrl.Rule(va['wysoki']  & ch['wysoki'],               quality['slabe']),
            ctrl.Rule(ch['wysoki']  & ca['niski'] & su['niski'],  quality['slabe']),
            ctrl.Rule(alc['wysoki'] & va['sredni'],               quality['srednie']),
            ctrl.Rule(alc['sredni'] & va['niski'],                quality['srednie']),
            ctrl.Rule(alc['sredni'] & va['sredni'],               quality['srednie']),
            ctrl.Rule(su['sredni']  & ca['sredni'],               quality['srednie']),
            ctrl.Rule(alc['niski']  & va['niski'] & su['wysoki'], quality['srednie']),
        ]

    cs  = ctrl.ControlSystem(rules)
    sim = ctrl.ControlSystemSimulation(cs)
    return sim, quality

def plot_membership_functions(wine_type, input_values: dict):
    """
    Pokazuje MF trzech najważniejszych cech + zaznacza wartość wejściową pionową linią.
    wine_type: 'white' lub 'red'
    input_values: słownik {nazwa_cechy: wartość}
    """
    if wine_type == "white":
        panels = [
            ("Alkohol (d=1.08)",      np.arange(8.0, 14.3, 0.1),
             [("niski",  "#3498db", fuzz.trapmf(np.arange(8.0,14.3,0.1), [8.0,8.0,9.0,9.6])),
              ("sredni", "#f39c12", fuzz.trimf (np.arange(8.0,14.3,0.1), [9.2,10.2,11.2])),
              ("wysoki", "#27ae60", fuzz.trapmf(np.arange(8.0,14.3,0.1), [10.4,11.0,14.3,14.3]))],
             input_values.get("alcohol")),
            ("Kwasowość lotna (d=−0.80)", np.arange(0.08, 1.15, 0.01),
             [("niska",   "#27ae60", fuzz.trapmf(np.arange(0.08,1.15,0.01), [0.08,0.08,0.20,0.26])),
              ("srednia", "#f39c12", fuzz.trimf (np.arange(0.08,1.15,0.01), [0.23,0.28,0.35])),
              ("wysoka",  "#e74c3c", fuzz.trapmf(np.arange(0.08,1.15,0.01), [0.32,0.40,1.15,1.15]))],
             input_values.get("volatile acidity")),
            ("Gęstość (d=−0.73)", np.arange(0.987, 1.039, 0.0001),
             [("niski",  "#27ae60", fuzz.trapmf(np.arange(0.987,1.039,0.0001), [0.987,0.987,0.9905,0.9930])),
              ("sredni", "#f39c12", fuzz.trimf (np.arange(0.987,1.039,0.0001), [0.9918,0.9942,0.9962])),
              ("wysoki", "#e74c3c", fuzz.trapmf(np.arange(0.987,1.039,0.0001), [0.9952,0.9972,1.039,1.039]))],
             input_values.get("density")),
        ]
    else:
        panels = [
            ("Alkohol (d=1.36)",      np.arange(8.4, 14.9, 0.1),
             [("niski",  "#3498db", fuzz.trapmf(np.arange(8.4,14.9,0.1), [8.4,8.4,9.4,9.9])),
              ("sredni", "#f39c12", fuzz.trimf (np.arange(8.4,14.9,0.1), [9.5,10.3,11.2])),
              ("wysoki", "#27ae60", fuzz.trapmf(np.arange(8.4,14.9,0.1), [10.8,11.6,14.9,14.9]))],
             input_values.get("alcohol")),
            ("Kwasowość lotna (d=−1.57)", np.arange(0.12, 1.58, 0.01),
             [("niski",  "#27ae60", fuzz.trapmf(np.arange(0.12,1.58,0.01), [0.12,0.12,0.32,0.42])),
              ("sredni", "#f39c12", fuzz.trimf (np.arange(0.12,1.58,0.01), [0.38,0.52,0.66])),
              ("wysoki", "#e74c3c", fuzz.trapmf(np.arange(0.12,1.58,0.01), [0.60,0.74,1.58,1.58]))],
             input_values.get("volatile acidity")),
            ("Siarczany (d=0.82)", np.arange(0.33, 2.0, 0.01),
             [("niski",  "#e74c3c", fuzz.trapmf(np.arange(0.33,2.0,0.01), [0.33,0.33,0.51,0.58])),
              ("sredni", "#f39c12", fuzz.trimf (np.arange(0.33,2.0,0.01), [0.54,0.64,0.75])),
              ("wysoki", "#27ae60", fuzz.trapmf(np.arange(0.33,2.0,0.01), [0.70,0.82,2.0,2.0]))],
             input_values.get("sulphates")),
        ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(
        f"Funkcje przynależności – wino {WINE_LABEL[wine_type]} (top-3 predyktory)",
        fontsize=13, fontweight='bold'
    )

    for ax, (title, universe, terms, val) in zip(axes, panels):
        for label, color, mf_vals in terms:
            ax.plot(universe, mf_vals, color=color, lw=2, label=label)
            ax.fill_between(universe, 0, mf_vals, color=color, alpha=0.10)
        if val is not None:
            ax.axvline(val, color='black', lw=2, ls='--', label=f'wartość={val:.4g}')
            for label, color, mf_vals in terms:
                mu = float(fuzz.interp_membership(universe, mf_vals, val))
                if mu > 0.01:
                    ax.plot(val, mu, 'o', color=color, ms=8, zorder=5)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Wartość cechy")
        ax.set_ylabel("μ")
        ax.set_ylim(-0.05, 1.15)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(alpha=0.3)

    plt.tight_layout()
    return fig

def plot_aggregated_output(sim, quality_obj, centroid_val, wine_type, input_values):
    universe  = quality_obj.universe
    mf_slabe  = quality_obj['slabe'].mf
    mf_sr     = quality_obj['srednie'].mf
    mf_dobre  = quality_obj['dobre'].mf

    aggregated = np.zeros_like(universe)
    term_activations = {}
    for tname, term in quality_obj.terms.items():
        act = getattr(term, '_activation', 0.0) or 0.0
        term_activations[tname] = act
        clipped = np.fmin(act, term.mf)
        aggregated = np.fmax(aggregated, clipped)

    final_class = 0 if centroid_val < 0.80 else (2 if centroid_val > 1.20 else 1)
    class_name  = CLASS_NAMES[final_class]
    class_color = CLASS_COLORS[class_name]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    fig.suptitle(
        f"Defuzyfikacja metodą środka ciężkości – wino {WINE_LABEL[wine_type]}\n"
        f"alc={input_values.get('alcohol','?'):.2g}  "
        f"va={input_values.get('volatile acidity','?'):.3g}  "
        f"den={input_values.get('density','?'):.5g}",
        fontsize=12, fontweight='bold'
    )

    ax1.plot(universe, mf_slabe,  color=CLASS_COLORS['slabe'],   lw=2.5, label='slabe')
    ax1.plot(universe, mf_sr,     color=CLASS_COLORS['srednie'], lw=2.5, label='średnie')
    ax1.plot(universe, mf_dobre,  color=CLASS_COLORS['dobre'],   lw=2.5, label='dobre')

    for tname, act in term_activations.items():
        col = CLASS_COLORS[tname]
        if act > 0.01:
            ax1.axhline(act, color=col, ls=':', lw=1.2, alpha=0.7)
            ax1.text(2.02, act, f'α={act:.2f}', color=col, fontsize=8, va='center')

    ax1.set_title("Oryginalne funkcje przynależności wyjścia", fontsize=10)
    ax1.set_ylabel("μ")
    ax1.set_ylim(-0.05, 1.20)
    ax1.legend(loc='upper center', ncol=3, fontsize=9)
    ax1.grid(alpha=0.3)

    ax2.fill_between(universe, 0, aggregated, alpha=0.30, color='steelblue')
    ax2.plot(universe, aggregated, color='steelblue', lw=2, label='Zagregowany obszar')

    ax2.plot(universe, mf_slabe, color=CLASS_COLORS['slabe'],   lw=1, ls='--', alpha=0.4)
    ax2.plot(universe, mf_sr,    color=CLASS_COLORS['srednie'], lw=1, ls='--', alpha=0.4)
    ax2.plot(universe, mf_dobre, color=CLASS_COLORS['dobre'],   lw=1, ls='--', alpha=0.4)

    ax2.fill_between(universe, 0, aggregated,
                     where=(universe <= centroid_val),
                     alpha=0.20, color='navy', label='Lewa połowa momentu')

    ax2.axvline(centroid_val, color='navy', lw=2.5,
                label=f'Centroid = {centroid_val:.4f}')

    offset = 0.25 if centroid_val < 1.5 else -0.25
    ax2.annotate(
        f'Centroid = {centroid_val:.4f}\n→ klasa: {class_name.upper()}',
        xy=(centroid_val, max(aggregated) * 0.5),
        xytext=(centroid_val + offset, max(aggregated) * 0.75 + 0.1),
        fontsize=10, fontweight='bold', color='navy',
        arrowprops=dict(arrowstyle='->', color='navy', lw=1.5),
        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', edgecolor='navy')
    )

    ax2.set_title("Zagregowany obszar aktywny po zastosowaniu reguł (Mamdani)", fontsize=10)
    ax2.set_ylabel("μ (po ucięciu min)")
    ax2.set_xlabel("Zmienna wyjściowa 'quality'")
    ax2.set_ylim(-0.05, 1.20)
    ax2.legend(loc='upper center', ncol=3, fontsize=9)
    ax2.grid(alpha=0.3)

    ax2.set_xticks([0, 0.5, 1.0, 1.5, 2.0])
    ax2.set_xticklabels(['0\n(slabe)', '0.5', '1.0\n(średnie)', '1.5', '2.0\n(dobre)'])

    plt.tight_layout()
    return fig

def plot_rule_activations(sim, wine_type, input_values):
    rules = sim.ctrl.rules

    labels = []
    strengths = []
    colors = []

    for i, rule in enumerate(rules):

        try:
            fs = rule.antecedent.membership_value(sim)
        except Exception:
            fs = 1.0

        cons = rule.consequent
        if isinstance(cons, list):
            term_name = cons[0].term.label
        else:
            term_name = cons.term.label

        labels.append(f"R{i+1} → {term_name}")
        strengths.append(fs)
        colors.append(CLASS_COLORS.get(term_name, "#95a5a6"))

    order = np.argsort(strengths)[::-1]
    labels = [labels[i] for i in order]
    strengths = [strengths[i] for i in order]
    colors = [colors[i] for i in order]

    fig, ax = plt.subplots(figsize=(10, max(5, len(labels) * 0.35)))

    bars = ax.barh(range(len(labels)), strengths, color=colors)

    for bar, val in zip(bars, strengths):
        ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                f"{val:.2f}", va="center", fontsize=8)

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Siła aktywacji reguły")
    ax.set_title("Aktywacja reguł (firing strength)")
    ax.grid(axis='x', alpha=0.3)
    ax.invert_yaxis()

    plt.tight_layout()
    return fig

def load_feature_names():
    return [
        "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
        "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
        "pH", "sulphates", "alcohol"
    ]

def ask_user_for_features(feature_names):
    print("\nPodaj wartości wszystkich 11 cech, oddzielone średnikami (;)")
    print("Kolejność:", "; ".join(feature_names))
    raw    = input("\nWartości: ").strip().replace(",", ".")
    values = [float(x.strip()) for x in raw.split(";")]
    if len(values) != len(feature_names):
        raise ValueError(f"Oczekiwano {len(feature_names)} wartości, podano {len(values)}.")
    return pd.DataFrame([values], columns=feature_names)

def main():
    print("=" * 56)
    print("  SYSTEM ROZMYTY MAMDANIEGO – OCENA JAKOŚCI WINA  v3")
    print("  Defuzyfikacja: środek ciężkości (centroid)")
    print("=" * 56)

    wine_type = input("\nWybierz typ wina (red/white): ").strip().lower()
    if wine_type not in ["red", "white"]:
        raise ValueError("Dozwolone opcje: 'red' lub 'white'.")

    feature_names = load_feature_names()
    X_user_raw    = ask_user_for_features(feature_names)
    input_values  = dict(zip(feature_names, X_user_raw.values[0]))

    sim, quality_obj = build_complete_fuzzy_system(wine_type)

    for feat in feature_names:
        fname = feat.replace(" ", "_")
        try:
            sim.input[fname] = input_values[feat]
        except (ValueError, KeyError):
            pass

    try:
        sim.compute()
        centroid    = sim.output['quality']
        final_class = 0 if centroid < 0.80 else (2 if centroid > 1.20 else 1)
        class_name  = CLASS_NAMES[final_class]

        print("\n" + "═" * 50)
        print("  WYNIK SYSTEMU EKSPERCKIEGO")
        print("═" * 50)
        print(f"  Wartość wyjściowa (centroid) : {centroid:.4f}")
        print(f"  Klasyfikacja                 : Klasa {final_class} ({class_name.upper()})")
        print("═" * 50)

        print("\nGenerowanie wykresów...")

        fig1 = plot_membership_functions(wine_type, input_values)
        fig2 = plot_aggregated_output(sim, quality_obj, centroid, wine_type, input_values)
        fig3 = plot_rule_activations(sim, wine_type, input_values)

        fig1.savefig("wykres1_MF_predyktorow.png",       dpi=150, bbox_inches='tight')
        fig2.savefig("wykres2_agregat_centroid.png",      dpi=150, bbox_inches='tight')
        fig3.savefig("wykres3_aktywacje_regul.png",       dpi=150, bbox_inches='tight')
        print("  → Zapisano: wykres1_MF_predyktorow.png")
        print("  → Zapisano: wykres2_agregat_centroid.png")
        print("  → Zapisano: wykres3_aktywacje_regul.png")

        plt.show()

    except Exception as e:
        print(f"\n[Błąd obliczeń]: {e}")
        print(f"Werdykt bezpieczny (Fallback): Klasa 1 ({CLASS_NAMES[1].upper()})")
        raise
def evaluate_system(file_path, wine_type, sim_builder):
    df = pd.read_csv(file_path, sep=';')

    y_true = []
    y_pred = []

    feature_names = load_feature_names()

    for _, row in df.iterrows():

        sim, _ = sim_builder(wine_type)

        for feat in feature_names:
            fname = feat.replace(" ", "_")
            try:
                sim.input[fname] = row[feat]
            except:
                pass

        try:
            sim.compute()
            score = sim.output['quality']

            if score < 0.80:
                pred = 0
            elif score > 1.20:
                pred = 2
            else:
                pred = 1

        except:
            pred = 1

        q = row['quality']
        if q <= 4:
            true = 0
        elif q <= 6:
            true = 1
        else:
            true = 2

        y_true.append(true)
        y_pred.append(pred)

    acc = accuracy_score(y_true, y_pred)
    cm  = confusion_matrix(y_true, y_pred)

    print("\n==============================")
    print(f"EWALUACJA: {wine_type.upper()}")
    print(f"Accuracy: {acc:.4f}")
    print("==============================")

    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=["slabe","srednie","dobre"],
                yticklabels=["slabe","srednie","dobre"],
                cmap="Blues")
    plt.title(f"Confusion Matrix - {wine_type}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()

    labels = ["slabe", "srednie", "dobre"]

    true_counts = np.bincount(y_true, minlength=3)
    pred_counts = np.bincount(y_pred, minlength=3)

    x = np.arange(3)

    plt.figure(figsize=(6,4))

    plt.bar(x, true_counts, label="True", color="#4C78A8")
    plt.bar(x, pred_counts, bottom=true_counts, label="Pred", color="#F58518", alpha=0.7)

    plt.xticks(x, labels)
    plt.title("Rozkład klas (stacked)")
    plt.legend()
    plt.grid(axis='y', alpha=0.3)

    plt.show()

    return acc, cm

if __name__ == "__main__":
    main()
    print("\nUruchamiam ewaluację na datasetach...")

    acc_r, cm_r = evaluate_system(
        "wine+quality/winequality-red.csv",
        "red",
        build_complete_fuzzy_system
    )

    acc_w, cm_w = evaluate_system(
        "wine+quality/winequality-white.csv",
        "white",
        build_complete_fuzzy_system
    )


