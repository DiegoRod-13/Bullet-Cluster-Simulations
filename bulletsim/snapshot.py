import h5py
import numpy as np


def get_masses(f, ptype):
    """Return the particle masses for one particle type."""

    group = f"PartType{ptype}"

    with h5py.File(f, "r") as h:
        if "Masses" in h[group]:
            return np.array(h[group]["Masses"])

        n = h["Header"].attrs["NumPart_ThisFile"][ptype]
        m = h["Header"].attrs["MassTable"][ptype]

        return m * np.ones(n)


def read_dm(file):
    """Read dark matter positions, masses, and velocities."""

    with h5py.File(file, "r") as f:
        pos = f["PartType1"]["Coordinates"][:]
        vel = f["PartType1"]["Velocities"][:]

    mass = get_masses(file, 1)

    return pos, mass, vel


def read_gas(file):
    """Read gas positions, masses, densities, and internal energies."""

    with h5py.File(file, "r") as f:
        gas = f["PartType0"]

        pos = gas["Coordinates"][:]
        rho = gas["Density"][:]
        u = gas["InternalEnergy"][:]

    mass = get_masses(file, 0)

    return pos, mass, rho, u


def xyz(pos, center):
    """Return centered x, y, and z coordinates."""

    x = pos[:, 0] - center[0]
    y = pos[:, 1] - center[1]
    z = pos[:, 2] - center[2]

    return x, y, z


def v3d(vel):
    """Return the three velocity components."""

    vx = vel[:, 0]
    vy = vel[:, 1]
    vz = vel[:, 2]

    return vx, vy, vz