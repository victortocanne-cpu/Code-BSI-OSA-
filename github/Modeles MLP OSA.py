import os
import numpy as np
import joblib
import pickle
import optuna
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf


tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True) if tf.config.list_physical_devices('GPU') else None


all_folders = [
    rf"C:\Users\tarda\Desktop\dataset{i}lin\\" for i in range(1, 10)
] + [r"C:\Users\tarda\Desktop\datatest1lin\\"]


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
                force = float(fname.replace("N.CSV", ""))
                spectrum = read_spectrum(os.path.join(folder, fname))
                X.append(spectrum)
                y.append(force)
            except:
                continue
    return np.array(X), np.array(y)


def objective(trial):
    n_layers = trial.suggest_int("n_layers", 1, 3)
    units = [trial.suggest_int(f"units_l{i+1}", 8, 128, step=8) for i in range(n_layers)]
    activ = trial.suggest_categorical("activation", ["relu", "tanh", "sigmoid"])
    dropouts = [trial.suggest_float(f"dropout_l{i+1}", 0.0, 0.5) for i in range(n_layers)]

    global_best_models = []
    mse_scores = []
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

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = Sequential()
        model.add(Input(shape=(X_train_scaled.shape[1],)))

        for u, d in zip(units, dropouts):
            model.add(Dense(u, activation=activ))
            if d > 0:
                model.add(Dropout(d))

        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')

        model.fit(X_train_scaled, y_train,
                  validation_split=0.1,
                  epochs=50,
                  batch_size=16,
                  callbacks=[EarlyStopping(patience=10, restore_best_weights=True)],
                  verbose=0)

        y_pred = model.predict(X_test_scaled).flatten()
        mse = mean_squared_error(y_test, y_pred)
        mse_scores.append(mse)


        global_best_models.append((model, scaler))

    avg_mse = np.mean(mse_scores)


    if avg_mse < objective.best_mse:
        objective.best_mse = avg_mse
        for idx, (m, s) in enumerate(global_best_models):
            m.save(f"best_model_fold{idx+1}.keras")
            joblib.dump(s, f"best_scaler_fold{idx+1}.save")

    return avg_mse

objective.best_mse = float('inf')


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=30)

print("\nMeilleur score moyen MSE:", study.best_value)
print("Paramètres optimaux:")
print(study.best_params)
