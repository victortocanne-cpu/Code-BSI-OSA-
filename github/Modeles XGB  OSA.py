import os
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib


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
    r"C:\Users\tarda\Desktop\dataset1lin\\",
    r"C:\Users\tarda\Desktop\dataset2lin\\",
    r"C:\Users\tarda\Desktop\dataset3lin\\",
    r"C:\Users\tarda\Desktop\dataset4lin\\",
    r"C:\Users\tarda\Desktop\dataset5lin\\",
    r"C:\Users\tarda\Desktop\dataset6lin\\",
    r"C:\Users\tarda\Desktop\dataset7lin\\",
    r"C:\Users\tarda\Desktop\dataset8lin\\",
    r"C:\Users\tarda\Desktop\dataset9lin\\",
    r"C:\Users\tarda\Desktop\datatest1lin\\"
]

def read_spectrum(file_path):
    with open(file_path, 'r') as f:
        data_started = False
        refl = []
        for line in f:
            if '[TRACE DATA]' in line:
                data_started = True
                continue
            if data_started and ',' in line:
                try:
                    _, val = line.strip().split(',')
                    refl.append(float(val))
                except:
                    continue
    return np.array(refl)

def load_dataset(folder):
    X, y = [], []
    for fname in sorted(os.listdir(folder)):
        if fname.endswith('.CSV') and 'N' in fname:
            try:
                force = float(fname.replace('N.CSV', ''))
                spectrum = read_spectrum(os.path.join(folder, fname))
                if len(spectrum) > 0:
                    X.append(spectrum)
                    y.append(force)
            except:
                continue
    return np.array(X), np.array(y)


target_len = None

for i in range(10):
    train_folders = [f for j, f in enumerate(all_folders) if j != i]
    test_folder = all_folders[i]


    X_train_list, y_train_list = [], []
    for folder in train_folders:
        X_part, y_part = load_dataset(folder)
        if target_len is None:
            target_len = min([len(x) for x in X_part])
        X_train_list.append(np.array([x[:target_len] for x in X_part]))
        y_train_list.append(y_part)

    X_train = np.vstack(X_train_list)
    y_train = np.hstack(y_train_list)


    X_test, y_test = load_dataset(test_folder)
    X_test = np.array([x[:target_len] for x in X_test])


    model = XGBRegressor(**best_params)
    model.fit(X_train, y_train)


    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"\n Test sur dataset {i+1}")
    print(f"➡️  MSE = {mse:.5f}, MAE = {mae:.5f}")


    model_name = f"model_leaveout_dataset{i+1}.save"
    joblib.dump(model, model_name)
