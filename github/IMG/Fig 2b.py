import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.interpolate import interp1d
from matplotlib.ticker import ScalarFormatter

folders = [fr"C:\\Users\\tarda\\Desktop\\dataset{i}lin" for i in range(1, 11)]
labels = [f"Dataset {i}" for i in range(1, 10)]
colors = plt.cm.tab10.colors

def read_csv_spectrum(file_path):
    wavelengths, reflectivity = [], []
    with open(file_path, 'r') as f:
        data_started = False
        for line in f:
            if '[TRACE DATA]' in line:
                data_started = True
                continue
            if data_started and ',' in line:
                try:
                    wl, ref = line.strip().split(',')
                    wavelengths.append(float(wl))
                    P_ref_mW = 10 ** (-27.3 / 10)
                    reflectivity.append((float(ref)) / P_ref_mW * 100)

                except:
                    continue
    return np.array(wavelengths), np.array(reflectivity)

all_wavelengths, all_spectres, all_forces = [], [], []

for folder in folders:
    csv_files = []
    for fname in os.listdir(folder):
        if fname.endswith(".CSV") and 'N' in fname:
            try:
                force = float(fname.replace('N.CSV', ''))
                csv_files.append((force, fname))
            except:
                continue
    csv_files.sort(key=lambda x: x[0])

    spectres, forces = [], []
    wavelengths = None
    for force, fname in csv_files:
        path = os.path.join(folder, fname)
        wl, refl = read_csv_spectrum(path)
        if wavelengths is None:
            wavelengths = wl
        spectres.append(refl)
        forces.append(force)

    all_wavelengths.append(wavelengths)
    all_spectres.append(np.array(spectres))
    all_forces.append(np.array(forces))

n_frames = min(len(s) for s in all_spectres)

fig, ax = plt.subplots(figsize=(12, 10))
lines = [ax.plot([], [], lw=2, color=colors[i % 10])[0] for i in range(1, 10)]
label_groups = [ax.text(0, 0, '', color=colors[i % 10], fontsize=16, weight='bold') for i in range(1, 10)]
connector_lines = [ax.plot([], [], color='black', lw=0.5)[0] for _ in range(9)]
title = ax.text(0.5, 1.05, "", transform=ax.transAxes, ha="center", fontsize=16)

min_y = min(s.min() for s in all_spectres)
max_y = max(s.max() for s in all_spectres)
min_x = min(all_wavelengths[0])
max_x = max(all_wavelengths[0])

ax.set_xlim(1553, 1556)
ax.set_ylim(min_y, 1)

ax.set_xlabel("Wavelength (nm)", fontsize=26)
ax.set_ylabel("Reflectivity (%)", fontsize=26)
ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
plt.tick_params(axis='both', labelsize=18)
ax.ticklabel_format(axis='y', style='plain')
ax.legend(labels, ncol=2, fontsize=13  )

paused = [False]
frame_index = {'i': 0}

y_offsets = [0.5, 1.2, 0.3, 1.0, 0.7, 1.4, 0.4, 0.9, 0.6]
x_offsets = [1.0, -1.0, 0.5, -1.5, 1 , -1.2, 0.8, -1.2, 0.9]

def init():
    for line in lines:
        line.set_data([], [])
    for label in label_groups:
        label.set_text("")
    for conn in connector_lines:
        conn.set_data([], [])
    title.set_text("")
    return lines + label_groups + connector_lines + [title]

def update(_):
    if paused[0]:
        return lines + label_groups + connector_lines + [title]

    i = frame_index['i']
    if i >= n_frames:
        return lines + label_groups + connector_lines + [title]

    label_positions = []
    min_label_spacing = (max_y - min_y) * 0.03

    for idx in range(9):
        line = lines[idx]
        wl = all_wavelengths[idx + 1]
        refl = all_spectres[idx + 1][i]
        line.set_data(wl, refl)

        x_label = wl[len(wl) // 2] + x_offsets[idx]
        y_label = refl[len(wl) // 2] + y_offsets[idx] * (max_y - min_y) * 0.05

        adjustment_attempts = 0
        while any(abs(y_label - other_y) < min_label_spacing for _, other_y in label_positions) and adjustment_attempts < 10:
            y_label += min_label_spacing
            adjustment_attempts += 1

        f_interp = interp1d(wl, refl, kind='cubic')
        wl_fine = np.linspace(wl.min(), wl.max(), 1000)
        refl_fine = f_interp(wl_fine)

        norm_x = (wl_fine - x_label) / (max_x - min_x)
        norm_y = (refl_fine - y_label) / (max_y - min_y)
        distances = np.sqrt(norm_x**2 + norm_y**2)
        closest_idx = np.argmin(distances)
        x_data = wl_fine[closest_idx]
        y_data = refl_fine[closest_idx]

        label_groups[idx].set_position((x_label, y_label))
        label_groups[idx].set_text(labels[idx])
        connector_lines[idx].set_data([x_data, x_label], [y_data, y_label])

        label_positions.append((x_label, y_label))

    force_str = " | ".join(f"{labels[j]}: {all_forces[j][i]:.1f} N" for j in range(9))
    print(force_str)
    title.set_text(f"Spectre {i+1}/{n_frames} - {force_str}")
    frame_index['i'] += 1
    return lines + label_groups + connector_lines + [title]

def on_key(event):
    key = event.key
    if key == ' ':
        paused[0] = not paused[0]
    elif key == 'left':
        frame_index['i'] = max(0, frame_index['i'] - 1)
    elif key == 'right':
        frame_index['i'] = min(n_frames - 1, frame_index['i'] + 1)
    elif key == 'r':
        frame_index['i'] = 0
    elif key == 's':
        fig.savefig(f"frame_{frame_index['i']:03d}.png", dpi=300)
        print(f"Saved frame_{frame_index['i']:03d}.png")

fig.canvas.mpl_connect('key_press_event', on_key)
ani = animation.FuncAnimation(fig, update, frames=np.arange(n_frames * 10),
                              init_func=init, blit=True, interval=200)

plt.tight_layout()
plt.show()
