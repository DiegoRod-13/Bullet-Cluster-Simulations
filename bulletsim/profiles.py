import numpy as np

from .snapshot import xyz, v3d


def density_profile(pos, mass, center=[0, 0, 0], bins=100, rmin=None, rmax=None):
    """Calculate a spherical density profile."""

    x, y, z = xyz(pos, center)
    radius = np.sqrt(x**2 + y**2 + z**2)

    edges = np.logspace(np.log10(rmin), np.log10(rmax), bins + 1)
    shell_mass, _ = np.histogram(radius, bins=edges, weights=mass)

    shell_volume = 4.0 * np.pi / 3.0 * (edges[1:]**3 - edges[:-1]**3)
    density = shell_mass / shell_volume
    radius_mid = 0.5 * (edges[1:] + edges[:-1])

    return radius_mid, density


def vel_profile(pos, vel, center=[0, 0, 0], bins=100, rmin=None, rmax=None):
    """Calculate the mean particle speed as a function of radius."""

    x, y, z = xyz(pos, center)
    radius = np.sqrt(x**2 + y**2 + z**2)

    vx, vy, vz = v3d(vel)
    speed = np.sqrt(vx**2 + vy**2 + vz**2)

    edges = np.logspace(np.log10(rmin), np.log10(rmax), bins + 1)
    speed_sum, _ = np.histogram(radius, bins=edges, weights=speed)
    counts, _ = np.histogram(radius, bins=edges)

    
    mean_speed = speed_sum / counts
    radius_mid = 0.5 * (edges[1:] + edges[:-1])

    return radius_mid, mean_speed