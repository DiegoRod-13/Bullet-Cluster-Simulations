from pathlib import Path

import matplotlib.pyplot as plt
from astropy.visualization import AsinhStretch, PercentileInterval

from bulletsim.plotting import plot_dm_contours, plot_dm_peaks
from bulletsim.xray import mock_chandra_map


files = [
    Path("data/model_initial.hdf5"),
    Path("data/model_elongated.hdf5"),
    Path("data/model_preferred.hdf5"),
]

titles = [
    "Initial geometry",
    r"Elongated bullet, $P=0.31$",
    "Preferred orientation",
]

view = (-864, 864, -864, 864)
ticks = [-800, -400, 0, 400, 800]
stretch = AsinhStretch(0.04) + PercentileInterval(99.5)

fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True, gridspec_kw={"wspace": 0, "hspace": 0})

for i, file in enumerate(files):
    ax = axes[i]

    image = mock_chandra_map(file, emin=0.5, emax=2.0, redshift=0.296, metallicity=0, box=1800, npix=800)

    ax.imshow(stretch(image), origin="lower", cmap="inferno", extent=view)
    plot_dm_contours(ax, file, view=view, bins=100, sigma=1, percentiles=(50, 70, 80, 95, 99.5))
    _, _, separation = plot_dm_peaks(ax, file, center=[0, 0, 0], bins=100, box=1800, sigma=3, threshold=0.1)

    ax.text(0.03, 0.96, titles[i], transform=ax.transAxes, color="white", fontsize=9, va="top")

    if separation is not None:
        ax.text(0.03, 0.87, f"$d={separation:.0f}$ kpc", transform=ax.transAxes, color="white", fontsize=8, va="top")

    ax.set(xlim=view[:2], ylim=view[2:], xticks=ticks, yticks=ticks, aspect="equal")
    ax.tick_params(bottom=True, labelbottom=True, left=i == 0, labelleft=i == 0, labelsize=9)

fig.supxlabel(r"$x\ [\mathrm{kpc}]$")
fig.supylabel(r"$y\ [\mathrm{kpc}]$")

fig.subplots_adjust(left=0.06, right=0.99, bottom=0.14, top=0.98, wspace=0, hspace=0)
fig.savefig("triaxial_comparison.pdf", dpi=300, bbox_inches="tight", facecolor="white")

plt.show()