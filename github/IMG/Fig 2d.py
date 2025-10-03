import os
import numpy as np
import matplotlib.pyplot as plt

base_path = r"C:\Users\tarda\Desktop\DATASETS BSI"
folders = [os.path.join(base_path, f"DATASET {i}", "DATA 1") for i in range(1, 51)]

target_filename = "50.0N.txt"

def read_txt_spectrum(file_path):
    df = np.genfromtxt(file_path, delimiter="\t", skip_header=1)
    wl = df[:, 0]
    refl = df[:, 1]
    return wl, refl

plt.figure(figsize=(14, 10))
colors = plt.cm.viridis(np.linspace(0, 1, len(folders)))
n_loaded = 0

for idx, folder in enumerate(folders):
    path = os.path.join(folder, target_filename)
    if os.path.exists(path):
        try:
            wl, refl = read_txt_spectrum(path)

            refl_norm = refl / np.max(refl)

            plt.plot(wl, refl_norm, color=colors[idx], label=f"Dataset {idx+1}")
            n_loaded += 1
        except Exception as e:
            print(f"Erreur dans {path} : {e}")
    else:
        print(f"Fichier manquant : {path}")

plt.xlabel("Wavelength (nm)", fontsize=30)
plt.ylabel(" Reflectivity (%) ", fontsize=30)
plt.tick_params(axis='both', labelsize=20)
plt.xlim(1550, 1560)
plt.ylim(0, 1)
plt.grid(False)
plt.tight_layout()
plt.savefig("nom_fichier2d.png", dpi=300, bbox_inches='tight')

plt.show()
