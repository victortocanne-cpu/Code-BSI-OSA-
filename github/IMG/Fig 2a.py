import pandas as pd
import matplotlib.pyplot as plt
import os
P_ref_mW = 10**(-27.3 / 10)


dossier = r"C:\Users\tarda\Desktop\datatest1lin"


fichiers = {
    "0N.CSV": "Load 0 N",
    "5N.CSV": "Load 5 N",
    "10N.CSV": "Load 10 N",
    "20N.CSV": "Load 20 N",
    "35N.CSV": "Load 35 N",
}


couleurs = {
    "Load 0 N": "blue",
    "Load 5 N": "black",
    "Load 10 N": "orange",
    "Load 20 N": "green",
    "Load 35 N": "red",
}


positions_labels = {
    "Load 0 N": (1554.55, (1.8e-5)/P_ref_mW * 100),
    "Load 5 N": (1554.6, (1.5e-5)/P_ref_mW * 100),
    "Load 10 N": (1554.65, (1e-5)/P_ref_mW * 100),
    "Load 20 N": (1554.8, (0.9e-5)/P_ref_mW * 100),
    "Load 35 N": (1555.0, (0.8e-5)/P_ref_mW * 100),
}


styles = {
    "Load 0 N": '-',
    "Load 5 N": '--',
    "Load 10 N": '-.',
    "Load 20 N": ':',
    "Load 35 N": '-',
}


plt.figure(figsize=(12, 8))


for nom_fichier, label in fichiers.items():
    chemin = os.path.join(dossier, nom_fichier)

    df = pd.read_csv(chemin, skiprows=1)
    df.columns = df.columns.str.strip()
    df = df[(df["Wavelength"] >= 1554) & (df["Wavelength"] <= 1555.5)]

    x = df["Wavelength"]

    P_ref_mW = 10**(-27.3 / 10)
    y = (df["Reflectivity_linear"]) / P_ref_mW * 100


    couleur = couleurs.get(label, None)
    style = styles.get(label, '-')
    plt.plot(x, y, color=couleur, linestyle=style)



    if label in positions_labels:
        x_val, y_val = positions_labels[label]

        plt.text(
            x_val, y_val,
            label,
            fontsize=18,
            color=couleur,
            ha='center',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, boxstyle='round,pad=0.2')
        )



plt.xlabel("Wavelength (nm)",fontsize=24)
plt.ylabel("Reflectivity (%)", fontsize=24)

plt.tick_params(axis='both', labelsize=18)
plt.xlim(1554, 1555.4)
plt.ylim(bottom=0)
plt.ylim(0, 1)

plt.grid(False)
plt.subplots_adjust(top=0.9, bottom=0.1)
plt.savefig("figure1.png", dpi=300)
plt.show()

