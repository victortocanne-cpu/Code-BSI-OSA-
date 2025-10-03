import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib.ticker import ScalarFormatter




dossier = r"C:\Users\tarda\Desktop\DATASETS BSI\DATASET 1\DATA 1"


fichiers = {
    "0.0N.txt": "Load 0.0 N",
    "5.0N.txt": "Load 5.0 N",
    "10.0N.txt": "Load 10.0 N",
    "20.0N.txt": "Load 20.0 N",
    "35.0N.txt": "Load 35.0 N",

}


couleurs = {
    "Load 0.0 N": "blue",
    "Load 5.0 N": "black",
    "Load 10.0 N": "orange",
    "Load 20.0 N": "green",
    "Load 35.0 N": "red",
}

styles = {
    "Load 0.0 N": '-',
    "Load 5.0 N": '--',
    "Load 10.0 N": '-.',
    "Load 20.0 N": ':',
    "Load 35.0 N": '-',
}



P_ref = 0.00000000003
P_ref_mW = 10 ** (-P_ref / 10)

plt.figure(figsize=(12, 8))

for nom_fichier, label in fichiers.items():
    chemin = os.path.join(dossier, nom_fichier)

    df = pd.read_csv(chemin, sep="\t")
    df.columns = df.columns.str.strip()

    if not {"Wavelength", "Reflectivity"}.issubset(df.columns):
        print(f"Colonnes attendues manquantes dans {nom_fichier}")
        continue


    df = df[(df["Wavelength"] >= 1550) & (df["Wavelength"] <= 1560)]

    x = df["Wavelength"]
    y = df["Reflectivity"] / P_ref_mW * 0.00002

    plt.plot(x, y, color=couleurs.get(label, "gray"), linestyle=styles.get(label, '-'))
    x_vals = x.values
    y_vals = y.values
    i_max = int(np.nanargmax(y_vals))
    x0, y0 = float(x_vals[i_max]), float(y_vals[i_max])

    midx = 0.5 * (float(x.min()) + float(x.max()))
    place_right = x0 <= midx
    dx = 10 if place_right else -10
    ha = "left" if place_right else "right"

    plt.annotate(
        label,
        xy=(x0, y0),
        xytext=(dx, 0), textcoords="offset points",
        ha=ha, va="center", fontsize=18,
        color=couleurs.get(label, "black"),
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
        arrowprops=dict(arrowstyle="-", lw=0.8, color=couleurs.get(label, "black")),
    )






plt.xlabel("Wavelength (nm)", fontsize=24)
plt.ylabel("Reflectivity (%)", fontsize=24)
plt.gca().yaxis.set_major_formatter(ScalarFormatter())
plt.ticklabel_format(axis='y', style='plain')
plt.xlim(1553, 1556)

plt.ylim(0, 1)
plt.grid(False)
plt.tick_params(axis='both', labelsize=20)

plt.tight_layout()
plt.savefig("nom_fichier.png", dpi=300, bbox_inches='tight')

plt.show()
