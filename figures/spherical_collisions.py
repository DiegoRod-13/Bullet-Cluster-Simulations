from pathlib import Path

import matplotlib.pyplot as plt
from astropy.visualization import AsinhStretch, PercentileInterval

from bulletsim.plotting import plot_dm_contours, plot_dm_peaks
from bulletsim.xray import mock_chandra_map


files = [
    [Path("data/velocity_low.hdf5"), Path("data/reference.hdf5"), Path("data/velocity_high.hdf5")],
    [Path("data/impact_low.hdf5"), Path("data/reference.hdf5"), Path("data/impact_high.hdf5")],
]

velocities = [250, 1000, 3000]
impacts = [0, 300, 600]

view = (-864, 864, -864, 864)
ticks = [-800, -400, 0, 400, 800]
stretch = AsinhStretch(0.04) + PercentileInterval(99.5)

fig, axes = plt.subplots(2, 3, figsize=(10, 6.7), sharex=True, sharey=True, gridspec_kw={"wspace": 0, "hspace": 0})

for i in range(2):
    for j in range(3):
        file = files[i][j]
        ax = axes[i, j]

        image = mock_chandra_map(file, emin=0.5, emax=2.0, redshift=0.296, metallicity=0, box=1800, npix=800)

        ax.imshow(stretch(image), origin="lower", cmap="inferno", extent=view)
        plot_dm_contours(ax, file, view=view, bins=100, sigma=1, percentiles=(50, 70, 80, 95, 99.5))
        _, _, separation = plot_dm_peaks(ax, file, center=[0, 0, 0], bins=100, box=1800, sigma=3, threshold=0.1)

        if separation is not None:
            ax.text(0.97, 0.96, f"$d={separation:.0f}$ kpc", transform=ax.transAxes, color="white", fontsize=7, ha="right", va="top")

        if i == 0:
            ax.text(0.97, 0.87, rf"$v_i={velocities[j]}$ km s$^{{-1}}$", transform=ax.transAxes, color="white", fontsize=7, ha="right", va="top")
            ax.text(0.97, 0.78, r"$b=300$ kpc", transform=ax.transAxes, color="white", fontsize=7, ha="right", va="top")

        if i == 1:
            ax.text(0.97, 0.87, r"$v_i=1000$ km s$^{-1}$", transform=ax.transAxes, color="white", fontsize=7, ha="right", va="top")
            ax.text(0.97, 0.78, rf"$b={impacts[j]}$ kpc", transform=ax.transAxes, color="white", fontsize=7, ha="right", va="top")

        ax.set(xlim=view[:2], ylim=view[2:], xticks=ticks, yticks=ticks, aspect="equal")
        ax.tick_params(bottom=i == 1, labelbottom=i == 1, left=j == 0, labelleft=j == 0, labelsize=8)

axes[0, 0].set_ylabel("Velocity variation\n$y$ [kpc]", fontsize=9)
axes[1, 0].set_ylabel("Impact-parameter variation\n$y$ [kpc]", fontsize=9)

fig.supxlabel(r"$x\ [\mathrm{kpc}]$", fontsize=10)

fig.savefig("spherical_collision.pdf", dpi=300, bbox_inches="tight", facecolor="white")
plt.show()