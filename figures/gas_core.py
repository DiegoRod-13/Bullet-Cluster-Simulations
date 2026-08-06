from pathlib import Path

import matplotlib.pyplot as plt
from astropy.visualization import AsinhStretch, PercentileInterval

from bulletsim.plotting import plot_dm_contours, plot_dm_peaks
from bulletsim.xray import mock_chandra_map


snapshots = [
    [Path("data/model_a1_b1.hdf5"), Path("data/model_a1_b2.hdf5"), Path("data/model_a1_b3.hdf5")],
    [Path("data/model_a2_b1.hdf5"), Path("data/model_a2_b2.hdf5"), Path("data/model_a2_b3.hdf5")],
    [Path("data/model_a3_b1.hdf5"), Path("data/model_a3_b2.hdf5"), Path("data/model_a3_b3.hdf5")],
]

row_labels = ["A1", "A2", "A3"]
column_labels = ["B1", "B2", "B3"]

view = (-864, 864, -864, 864)
ticks = [-800, -400, 0, 400, 800]
stretch = AsinhStretch(0.04) + PercentileInterval(99.5)

fig, axes = plt.subplots(3, 3, figsize=(10, 10), sharex=True, sharey=True, gridspec_kw={"wspace": 0, "hspace": 0})

for i in range(3):
    for j in range(3):
        file = snapshots[i][j]
        ax = axes[i, j]

        image = mock_chandra_map(file, emin=0.5, emax=2.0, redshift=0.296, metallicity=0, box=1800, npix=800)

        ax.imshow(stretch(image), origin="lower", cmap="inferno", extent=view)

        plot_dm_contours(ax, file, view=view, bins=100, sigma=1, percentiles=(50, 70, 80, 95, 99.5))

        c1, c2, separation = plot_dm_peaks(ax, file, center=[0, 0, 0], bins=100, box=1800, sigma=3, threshold=0.1)

        if separation is not None:
            ax.text(0.03, 0.96, f"d = {separation:.0f} kpc", transform=ax.transAxes, color="white", fontsize=8, va="top")

        ax.set(xlim=view[:2], ylim=view[2:], xticks=ticks, yticks=ticks, aspect="equal")
        ax.tick_params(bottom=i == 2, labelbottom=i == 2, left=j == 0, labelleft=j == 0, labelsize=9)

        if i == 0:
            ax.set_title(column_labels[j])

        if j == 0:
            ax.set_ylabel(row_labels[i], rotation=0, labelpad=25, va="center")

fig.supxlabel(r"$x\ [\mathrm{kpc}]$")
fig.supylabel(r"$y\ [\mathrm{kpc}]$")

fig.savefig("parameter_grid.pdf", dpi=300, bbox_inches="tight", facecolor="white")
plt.show()