from pathlib import Path

import matplotlib.pyplot as plt

from bulletsim import read_dm, read_gas
from bulletsim.plotting import density_profile


main_files = [
    Path("data/main_000.hdf5"),
    Path("data/main_010.hdf5"),
    Path("data/main_020.hdf5"),
]

bullet_files = [
    Path("data/bullet_000.hdf5"),
    Path("data/bullet_010.hdf5"),
    Path("data/bullet_020.hdf5"),
]

times = [0.0, 1.0, 2.0]
colors = ["navy", "olive", "violet"]

main_center = [0, 0, 0]
bullet_center = [0, 0, 0]

main_r200 = 2140.13
bullet_r200 = 988.74

fig, axes = plt.subplots(2, 2, figsize=(7, 7), sharex=True, gridspec_kw={"wspace": 0.08, "hspace": 0.08})

for i, file in enumerate(main_files):
    pos, mass, vel = read_dm(file)
    r, rho = density_profile(pos, mass, center=main_center, bins=50, rmin=20, rmax=3000)
    axes[0, 0].loglog(r, rho, color=colors[i], label=rf"$t={times[i]:.1f}$ Gyr")

    pos, mass, density, u = read_gas(file)
    r, rho = density_profile(pos, mass, center=main_center, bins=50, rmin=20, rmax=3000)
    axes[0, 1].loglog(r, rho, color=colors[i])

for i, file in enumerate(bullet_files):
    pos, mass, vel = read_dm(file)
    r, rho = density_profile(pos, mass, center=bullet_center, bins=50, rmin=20, rmax=2000)
    axes[1, 0].loglog(r, rho, color=colors[i])

    pos, mass, density, u = read_gas(file)
    r, rho = density_profile(pos, mass, center=bullet_center, bins=50, rmin=20, rmax=2000)
    axes[1, 1].loglog(r, rho, color=colors[i])

axes[0, 0].axvline(main_r200, color="black", linestyle=":", linewidth=1)
axes[0, 1].axvline(main_r200, color="black", linestyle=":", linewidth=1)
axes[1, 0].axvline(bullet_r200, color="black", linestyle=":", linewidth=1)
axes[1, 1].axvline(bullet_r200, color="black", linestyle=":", linewidth=1)

axes[0, 0].set_title("Dark matter")
axes[0, 1].set_title("Gas")

axes[0, 0].set_ylabel("Main\n" + r"$\rho\ [M_\odot\,\mathrm{kpc}^{-3}]$")
axes[1, 0].set_ylabel("Bullet\n" + r"$\rho\ [M_\odot\,\mathrm{kpc}^{-3}]$")

axes[1, 0].set_xlabel(r"$r\ [\mathrm{kpc}]$")
axes[1, 1].set_xlabel(r"$r\ [\mathrm{kpc}]$")

axes[0, 0].legend(frameon=False, fontsize=8)

for ax in axes.flat:
    ax.grid(alpha=0.25)

fig.savefig("spherical_stability.pdf", dpi=300, bbox_inches="tight", facecolor="white")
plt.show()