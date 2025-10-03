
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
import joblib

def normalize_per_spectrum(s):
    s = np.array(s)
    mean = np.mean(s)
    std = np.std(s)
    return (s - mean) / std if std > 0 else s - mean

base_root = os.path.join(os.path.expanduser("~"), "Desktop")
dataset_roots = [os.path.join(base_root, "DATASETS BSI", f"DATASET {i}") for i in range(1, 51)]

all_datasets = []

print("📥 Chargement des 50 datasets...")
for dataset_root in dataset_roots:
    folder = os.path.join(dataset_root, "DATA 1")
    if not os.path.isdir(folder):
        print(f" Dossier manquant : {folder}")
        continue

    X_list, y_list = [], []
    for fname in sorted(os.listdir(folder)):
        if fname.endswith("N.txt"):
            try:
                force = float(fname.replace("N.txt", ""))
                path = os.path.join(folder, fname)
                df = pd.read_csv(path, sep=None, engine='python')
                df = df.sort_values(by=df.columns[0])
                reflectivity = df.iloc[:, 1].values
                reflectivity_norm = normalize_per_spectrum(reflectivity)
                X_list.append(reflectivity_norm)
                y_list.append(force)
            except Exception as e:
                print(f"⚠️ Erreur dans {fname} : {e}")
    if X_list:
        all_datasets.append((np.array(X_list), np.array(y_list)))
    else:
        print(f"⚠️ Aucun spectre trouvé dans {folder}")

print(f" {len(all_datasets)} datasets chargés.")

if len(all_datasets) < 2:
    print(" Pas assez de datasets pour Leave-One-Out. Arrêt.")
    exit()

output_dir = os.path.join(os.getcwd(), "loo_models")
os.makedirs(output_dir, exist_ok=True)

print("\n Lancement du Leave-One-Dataset-Out...")

for i in range(len(all_datasets)):
    print(f" Modèle {i+1}/50 : exclusion de DATASET {i+1}")


    X_train = np.vstack([X for j, (X, _) in enumerate(all_datasets) if j != i])
    y_train = np.hstack([y for j, (_, y) in enumerate(all_datasets) if j != i])

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    model.fit(
        X_train_scaled, y_train,
        validation_split=0.1,
        epochs=200,
        batch_size=16,
        verbose=0,
        callbacks=[EarlyStopping(patience=15, restore_best_weights=True)]
    )

    model_path = os.path.join(output_dir, f"model_{i+1:02d}.keras")
    scaler_path = os.path.join(output_dir, f"scaler_{i+1:02d}.save")
    model.save(model_path)
    joblib.dump(scaler, scaler_path)

    print(f"    Modèle {i+1:02d} sauvegardé.")

print("\n Tous les 50 modèles ont été entraînés et sauvegardés.")
