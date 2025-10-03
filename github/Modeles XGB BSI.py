
import os
import re
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib

# ================== HYPERPARAMÈTRES XGB ==================
best_params = {
    "learning_rate": 0.06347,
    "n_estimators": 512,
    "max_depth": 6,
    "reg_lambda": 1.97715,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "objective": "reg:squarederror",
    "random_state": 42,
    "n_jobs": 1,
    "verbosity": 0
}

all_folders = [
    r"C:\Users\tarda\Desktop\DATASETS BSI\DATASET 1\DATA 1\\",
    r"C:\Users\tarda\Desktop\DATASETS BSI\DATASET 2\DATA 1\\",
    r"C:\Users\tarda\Desktop\DATASETS BSI\DATASET 3\DATA 1\\",
    r"C:\Users\tarda\Desktop\DATASETS BSI\DATASET 4\DATA 1\\",
    r"C:\Users\tarda\Desktop\DATASETS BSI\DATASET 5\DATA 1\\",
    r"C:\Users\tarda\Desktop\DATASETS BSI\DATASET 6\DATA 1\\",
    r"C:\Users\tarda\Desktop\DATASETS BSI\DATASET 7\DATA 1\\",
    r"C:\Users\tarda\Desktop\DATASETS BSI\DATASET 8\DATA 1\\",
    r"C:\Users\tarda\Desktop\DATASETS BSI\DATASET 9\DATA 1\\",
    r"C:\Users\tarda\Desktop\DATASETS BSI\DATASET 10\DATA 1\\",
    r"C:\Users\tarda\Desktop\DATASETS BSI\DATASET 11\DATA 1\\",
    r"C:\Users\tarda\Desktop\DATASETS BSI\DATASET 12\DATA 1\\",
    r"C:\Users\tarda\Desktop\DATASETS BSI\DATASET 13\DATA 1\\",
    r"C:\Users\tarda\Desktop\DATASETS BSI\DATASET 14\DATA 1\\",
    r"C:\Users\tarda\Desktop\DATASETS BSI\DATASET 15\DATA 1\\",
    r"C:\Users\tarda\Desktop\DATASETS BSI\DATASET 16\DATA 1\\",
    r"C:\Users\tarda\Desktop\DATASETS BSI\DATASET 17\DATA 1\\",
    r"C:\Users\tarda\Desktop\DATASETS BSI\DATASET 18\DATA 1\\",
    r"C:\Users\tarda\Desktop\DATASETS BSI\DATASET 19\DATA 1\\",
    r"C:\Users\tarda\Desktop\DATASETS BSI\DATASET 20\DATA 1\\",
    r"C:\Users\tarda\Desktop\DATASETS BSI\DATASET 21\DATA 1\\",
    r"C:\Users\tarda\Desktop\DATASETS BSI\DATASET 22\DATA 1\\",
    r"C:\Users\tarda\Desktop\DATASETS BSI\DATASET 23\DATA 1\\",
    r"C:\Users\tarda\Desktop\DATASETS BSI\DATASET 24\DATA 1\\",
    r"C:\Users\tarda\Desktop\DATASETS BSI\DATASET 25\DATA 1\\",
    r"C:\Users\tarda\Desktop\DATASETS BSI\DATASET 26\DATA 1\\",
    r"C:\Users\tarda\Desktop\DATASETS BSI\DATASET 27\DATA 1\\",
    r"C:\Users\tarda\Desktop\DATASETS BSI\DATASET 28\DATA 1\\",
    r"C:\Users\tarda\Desktop\DATASETS BSI\DATASET 29\DATA 1\\",
    r"C:\Users\tarda\Desktop\DATASETS BSI\DATASET 30\DATA 1\\",
    r"C:\Users\tarda\Desktop\DATASETS BSI\DATASET 31\DATA 1\\",
    r"C:\Users\tarda\Desktop\DATASETS BSI\DATASET 32\DATA 1\\",
    r"C:\Users\tarda\Desktop\DATASETS BSI\DATASET 33\DATA 1\\",
    r"C:\Users\tarda\Desktop\DATASETS BSI\DATASET 34\DATA 1\\",
    r"C:\Users\tarda\Desktop\DATASETS BSI\DATASET 35\DATA 1\\",
    r"C:\Users\tarda\Desktop\DATASETS BSI\DATASET 36\DATA 1\\",
    r"C:\Users\tarda\Desktop\DATASETS BSI\DATASET 37\DATA 1\\",
    r"C:\Users\tarda\Desktop\DATASETS BSI\DATASET 38\DATA 1\\",
    r"C:\Users\tarda\Desktop\DATASETS BSI\DATASET 39\DATA 1\\",
    r"C:\Users\tarda\Desktop\DATASETS BSI\DATASET 40\DATA 1\\",
    r"C:\Users\tarda\Desktop\DATASETS BSI\DATASET 41\DATA 1\\",
    r"C:\Users\tarda\Desktop\DATASETS BSI\DATASET 42\DATA 1\\",
    r"C:\Users\tarda\Desktop\DATASETS BSI\DATASET 43\DATA 1\\",
    r"C:\Users\tarda\Desktop\DATASETS BSI\DATASET 44\DATA 1\\",
    r"C:\Users\tarda\Desktop\DATASETS BSI\DATASET 45\DATA 1\\",
    r"C:\Users\tarda\Desktop\DATASETS BSI\DATASET 46\DATA 1\\",
    r"C:\Users\tarda\Desktop\DATASETS BSI\DATASET 47\DATA 1\\",
    r"C:\Users\tarda\Desktop\DATASETS BSI\DATASET 48\DATA 1\\",
    r"C:\Users\tarda\Desktop\DATASETS BSI\DATASET 49\DATA 1\\",
    r"C:\Users\tarda\Desktop\DATASETS BSI\DATASET 50\DATA 1\\",
]

_force_re = re.compile(r'([0-9]+(?:\.[0-9]+)?)\s*N(?:\.[A-Za-z0-9]+)?$', re.IGNORECASE)
def parse_force_from_filename(fname: str):
    m = _force_re.search(fname)
    if not m:
        m = re.search(r'([0-9]+(?:\.[0-9]+)?)\s*N', fname, flags=re.IGNORECASE)
    return float(m.group(1)) if m else None


def read_spectrum(file_path: str) -> np.ndarray:
    vals = []
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if not any(c.isdigit() for c in line):
                continue
            parts = re.split(r'[,\t; ]+', line.strip())
            if len(parts) < 2:
                continue

            p0 = parts[0].lower()
            p1 = parts[1].lower()
            if p0.startswith('wavelength') or p1.startswith('reflect'):
                continue
            try:
                vals.append(float(parts[1]))
            except ValueError:
                continue
    return np.asarray(vals, dtype=float)


def load_dataset(folder: str):
    X, y = [], []
    for fname in sorted(os.listdir(folder)):
        up = fname.upper()
        if not (up.endswith('.TXT') or up.endswith('.CSV')):
            continue
        if 'N' not in up:
            continue
        force = parse_force_from_filename(fname)
        if force is None:
            continue
        spec = read_spectrum(os.path.join(folder, fname))
        if spec.size > 0:
            X.append(spec)
            y.append(force)
    return np.array(X, dtype=object), np.array(y, dtype=float)


print(f"[INFO] {len(all_folders)} datasets détectés. 1 modèle par fold.")
for i in range(len(all_folders)):
    train_folders = [f for j, f in enumerate(all_folders) if j != i]
    test_folder   = all_folders[i]


    X_train_raw, y_train_parts = [], []
    for folder in train_folders:
        X_part, y_part = load_dataset(folder)
        if X_part.size == 0:
            continue
        X_train_raw.extend(list(X_part))
        y_train_parts.append(y_part)

    X_test_raw, y_test = load_dataset(test_folder)


    if len(X_train_raw) == 0 or X_test_raw.size == 0:
        print(f"[WARN] Fold {i+1}: données insuffisantes (train={len(X_train_raw)}, test={X_test_raw.size}) → skip")
        continue


    all_series = X_train_raw + list(X_test_raw)
    lengths = [len(a) for a in all_series if len(a) > 0]
    target_len = min(lengths)

    X_train = np.vstack([a[:target_len] for a in X_train_raw])
    y_train = np.hstack(y_train_parts)
    X_test  = np.vstack([a[:target_len] for a in X_test_raw])


    model = XGBRegressor(**best_params)
    model.fit(X_train, y_train)


    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"[{i+1:02d}/{len(all_folders)}] Test={os.path.basename(os.path.dirname(test_folder))} "
          f"| train_files={len(y_train)} | test_files={len(y_test)} "
          f"| MSE={mse:.5f} | MAE={mae:.5f}")


    model_name = f"model_leaveout_dataset{i+1:02d}.save"
    joblib.dump(model, model_name)


