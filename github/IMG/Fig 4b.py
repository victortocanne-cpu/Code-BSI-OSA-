

import os, re
import numpy as np
import matplotlib.pyplot as plt
import joblib, pickle
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error


dossier_spectres = r"C:\Users\tarda\Desktop\DATASETS BSI\DATASET 1\DATA 1"
modele_mlp_path  = r"C:\Users\tarda\Desktop\BSI MODELE LEARNING\loo_models\model_01.keras"
scaler_mlp_path  = r"C:\Users\tarda\Desktop\BSI MODELE LEARNING\loo_models\scaler_01.save"
modele_xgb_path  = r"C:\Users\tarda\Desktop\BSI MODELE LEARNING\model_leaveout_dataset01.save"

ref_csv_path     = r"C:\Users\tarda\Desktop\BSI MODELE LEARNING\predictions_ref.csv"
out_png          = "comparaison_mlp_xgb_ref.png"



def read_spectrum_txt(filepath):
    """
    Lit un fichier .txt/.csv avec en-tête 'Wavelength  Reflectivity'.
    Retourne la colonne Reflectivity (float).
    """
    data = []
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        first = True
        for line in f:
            if first:
                first = False
                continue
            if not any(c.isdigit() for c in line):
                continue
            parts = re.split(r"[,\t; ]+", line.strip())
            if len(parts) >= 2:
                try:
                    data.append(float(parts[1]))
                except ValueError:
                    pass
    return np.asarray(data, dtype=float)


_force_re = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*N", re.IGNORECASE)
def get_label_from_filename(filename):
    m = _force_re.search(filename)
    return float(m.group(1)) if m else None

def align_length(v, L):
    v = np.asarray(v, dtype=float)
    if v.size == L: return v
    if v.size > L:  return v[:L]
    pad = np.full(L - v.size, v[-1] if v.size else 0.0, dtype=float)
    return np.concatenate([v, pad])

# ----------- Charger modèles -----------
mlp_model = load_model(modele_mlp_path)
scaler_mlp = joblib.load(scaler_mlp_path)

xgb_model = None
if os.path.isfile(modele_xgb_path):
    try:
        with open(modele_xgb_path, "rb") as f:
            xgb_model = pickle.load(f)
    except Exception as e:
        print(f"[INFO] XGB non chargé: {e}")

in_len = int(mlp_model.input_shape[1])

# ----------- Prédictions -----------
y_true, y_pred_mlp, y_pred_xgb = [], [], []

for fname in sorted(os.listdir(dossier_spectres)):
    up = fname.upper()
    if not (("N" in up) and (up.endswith(".TXT") or up.endswith(".CSV"))):
        continue

    label = get_label_from_filename(fname)
    if label is None:
        print(f"Skip (pas de 'N' parsable): {fname}")
        continue

    path = os.path.join(dossier_spectres, fname)
    try:
        spec = read_spectrum_txt(path)
        if spec.size == 0:
            print(f"Skip (spectre vide): {fname}")
            continue



        spec_mlp = align_length(spec, in_len)
        X_mlp = scaler_mlp.transform([spec_mlp])
        pred_mlp = float(mlp_model.predict(X_mlp, verbose=0)[0, 0])


        pred_xgb = None
        if xgb_model is not None:
            spec_xgb = align_length(spec, in_len)
            pred_xgb = float(xgb_model.predict([spec_xgb])[0])

        y_true.append(label)
        y_pred_mlp.append(pred_mlp)
        if pred_xgb is not None:
            y_pred_xgb.append(pred_xgb)

    except Exception as e:
        print(f"[ERR] {fname}: {e}")

if len(y_true) == 0:
    raise ValueError("Aucun échantillon valide trouvé. Vérifie le dossier et les formats.")

# ----------- Métriques -----------
mae_mlp = mean_absolute_error(y_true, y_pred_mlp)
mse_mlp = mean_squared_error(y_true, y_pred_mlp)
print(f"MLP     → MAE = {mae_mlp:.3f} N, MSE = {mse_mlp:.3f} N")

mae_xgb = mse_xgb = None
if len(y_pred_xgb) == len(y_true) and len(y_pred_xgb) > 0:
    mae_xgb = mean_absolute_error(y_true, y_pred_xgb)
    mse_xgb = mean_squared_error(y_true, y_pred_xgb)
    print(f"XGBoost → MAE = {mae_xgb:.3f} N, MSE = {mse_xgb:.3f} N")

# ----------- Tracé -----------
plt.figure(figsize=(8,6))
plt.plot(y_true, y_true, 'r-', label="target curve (y = x)", zorder=1)



if mae_xgb is not None:
    plt.scatter(y_true, y_pred_xgb, s=24, color='green', marker='x',
                label=f"XGB (MAE={mae_xgb:.2f}, MSE={mse_xgb:.2f})", alpha=0.8)


if os.path.isfile(ref_csv_path):
    try:
        ref = pd.read_csv(ref_csv_path)
        if {"true","pred"}.issubset(ref.columns):
            ref = ref.sort_values(by="true")
            plt.scatter(ref["true"], ref["pred"], s=28, color='blue', marker='o',
                        alpha=0.8, label="MLP (MAE=2.80, MSE=9.59)")
        else:
            print("⚠️ predictions_ref.csv ne contient pas les colonnes true,pred")
    except Exception as e:
        print(f"⚠️ Impossible de lire la courbe de référence: {e}")
else:
    print(f"⚠️ Fichier de référence introuvable : {ref_csv_path}")

plt.xlabel("True Load (N)", fontsize=20)
plt.ylabel("Predicted Load (N)", fontsize=20)
plt.xlim(0, 70); plt.ylim(0, 70)
plt.grid(False)

handles, labels = plt.gca().get_legend_handles_labels()


order = [0, 2, 1]


plt.legend([handles[i] for i in order], [labels[i] for i in order], fontsize=18)


plt.tick_params(axis='both', labelsize=18)

plt.tight_layout()

# ----------- AJOUT DIAGRAMME MAE GLOBAL -----------

models = ['MLP', 'XGBoost']
mae_vals = [4.816, 0.9153]
std_vals = [2.281, 0.6499]

inset_ax = plt.axes([0.65, 0.2, 0.3, 0.3])  # [left, bottom, width, height]
inset_ax.bar(models, mae_vals, yerr=std_vals, capsize=5,
             color=['blue', 'green'], alpha=0.8)
inset_ax.set_title("Global MAE ± STD", fontsize=12)
inset_ax.set_ylabel("MAE (N)", fontsize=12)
inset_ax.tick_params(axis='both', labelsize=12)
inset_ax.set_ylim(0, max(mae_vals[i] + std_vals[i] for i in range(2)) + 1 )
inset_ax.grid(True,linestyle='--', alpha=0.4)

plt.savefig("comparaison_mlp_dtr.png", dpi=300)
plt.show()

