import os
import numpy as np
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle


dossier_spectres = r"C:\Users\tarda\Desktop\dataset10lin"
modele_mlp_path = r"C:\Users\tarda\Desktop\Optimisateurs\mlp leave one out\best_model_fold10.keras"
scaler_mlp_path = r"C:\Users\tarda\Desktop\Optimisateurs\mlp leave one out\best_scaler_fold10.save"
modele_xgb_path = r"C:\Users\tarda\Desktop\reseauXGB\modeles xgb\model_leaveout_dataset10.save"


def read_spectrum_raw(file_path):
    with open(file_path, 'r') as f:
        data_started = False
        refl = []
        for line in f:
            if '[TRACE DATA]' in line:
                data_started = True
                continue
            if data_started and ',' in line and any(c.isdigit() for c in line):
                try:
                    _, ref = line.strip().split(',')
                    refl.append(float(ref))
                except:
                    continue
    return np.array(refl)


mlp_model = load_model(modele_mlp_path)
scaler_mlp = joblib.load(scaler_mlp_path)

with open(modele_xgb_path, 'rb') as f:
    xgb_model = pickle.load(f)


y_true, y_pred_mlp, y_pred_xgb = [], [], []

for fname in sorted(os.listdir(dossier_spectres)):
    if fname.endswith('.CSV') and 'N' in fname:
        try:
            label = float(fname.replace('N.CSV',''))
            path = os.path.join(dossier_spectres, fname)
            spectre = read_spectrum_raw(path)
            spectre = spectre[:mlp_model.input_shape[1]]

            scaled_mlp = scaler_mlp.transform([spectre])
            pred_mlp = mlp_model.predict(scaled_mlp)[0, 0]
            pred_xgb = xgb_model.predict([spectre])[0]

            y_true.append(label)
            y_pred_mlp.append(pred_mlp)
            y_pred_xgb.append(pred_xgb)

        except Exception as e:
            print(f"Erreur {fname}: {e}")



mae_mlp = mean_absolute_error(y_true, y_pred_mlp)
mse_mlp = mean_squared_error(y_true, y_pred_mlp)

mae_xgb = mean_absolute_error(y_true, y_pred_xgb)
mse_xgb = mean_squared_error(y_true, y_pred_xgb)

print(f"MLP     → MAE = {mae_mlp:.2f} N, MSE = {mse_mlp:.2f} N")
print(f"XGBoost → MAE = {mae_xgb:.2f} N, MSE = {mse_xgb:.2f} N")



plt.figure(figsize=(8,6))

plt.plot(y_true, y_true, 'r-', label="target curve (y = x)",zorder=1)
plt.scatter(y_true, y_pred_mlp, s=30, color='blue', marker='o',
            label=f"MLP (MAE={mae_mlp:.2f}, MSE={mse_mlp:.2f})", alpha=0.7)
plt.scatter(y_true, y_pred_xgb, s=30, color='green', marker='x',
            label=f"XGB (MAE={mae_xgb:.2f}, MSE={mse_xgb:.2f})", alpha=0.7)

plt.xlabel("True Load (N)",fontsize=20)
plt.ylabel("Prediction Load (N)",fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.xlim(0, 70)
plt.ylim(0, 70)

plt.legend(fontsize=16)
plt.grid(False)
plt.tight_layout()



models = ['MLP', 'XGBoost']
mae_vals = [2.63, 1.44]
std_vals = [1.09, 1.10]

inset_ax = plt.axes([0.65, 0.2, 0.3, 0.3])
inset_ax.bar(models, mae_vals, yerr=std_vals, capsize=5,
             color=['blue', 'green'], alpha=0.8)
inset_ax.set_title("Global MAE ± STD", fontsize=12)
inset_ax.set_ylabel("MAE (N)", fontsize=12)
inset_ax.tick_params(axis='both', labelsize=12)
inset_ax.set_ylim(0, 8)

inset_ax.grid(True, linestyle='--', alpha=0.4)

plt.savefig("comparaison_mlp_dtr.png", dpi=300)
plt.show()
