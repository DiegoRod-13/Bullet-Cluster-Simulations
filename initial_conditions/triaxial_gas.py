import agama
import h5py
import numpy as np

from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.optimize import brentq


MUNIT = 1e10

agama.setUnits(length=1, velocity=1, mass=MUNIT)

G = 4.30091727003628e-6 * MUNIT

z = 0.296
H0 = 70.0
OmegaM = 0.3
OmegaL = 0.7

fgas = 0.13
rng = np.random.default_rng(42)


def rho_crit_z(z):
    Hz = H0 * np.sqrt(OmegaM * (1 + z)**3 + OmegaL)
    Hz = Hz / 1000.0
    return 3 * Hz**2 / (8 * np.pi * G)


def r200_from_m200(M200, z):
    rho_c = rho_crit_z(z)
    return (3 * M200 / (4 * np.pi * 200 * rho_c))**(1 / 3)


def enclosed_mass_ellipsoid(density, radius, P, Q):
    def integrand(R):
        return density.density([R, 0, 0]) * R**2

    integral = quad(integrand, 0, radius, epsabs=0, epsrel=1e-6)[0]
    return 4 * np.pi * P * Q * integral


def density_norm_for_m200(M200, R200, rs, P, Q):
    density = agama.Density(type="Spheroid", axisRatioY=Q, axisRatioZ=P, scaleRadius=rs, densityNorm=1.0, alpha=1, beta=3, gamma=1, outerCutoffRadius=2 * R200, cutoffStrength=8)
    mass_unit = enclosed_mass_ellipsoid(density, R200, P, Q)
    return M200 / mass_unit


# Construct the continuous triaxial dark matter model.
def make_dm_potential(M200, concentration, P, Q):
    R200 = r200_from_m200(M200, z)
    rs = R200 / concentration
    density_norm = density_norm_for_m200(M200, R200, rs, P, Q)

    density = agama.Density(type="Spheroid", axisRatioY=Q, axisRatioZ=P, scaleRadius=rs, densityNorm=density_norm, alpha=1, beta=3, gamma=1, outerCutoffRadius=2 * R200, cutoffStrength=8)
    potential = agama.Potential(type="Multipole", lmax=8, mmax=6, density=density)

    return R200, rs, potential


# Beta-model gas density.
def gas_density(R, rho0, rc, beta):
    return rho0 * (1 + (R / rc)**2)**(-1.5 * beta)


# Tabulate the axis ratios of the equipotential surfaces.
def make_pq_lookup(potential, Rmin, Rmax, n=1000):
    Rgrid = np.logspace(np.log10(Rmin), np.log10(Rmax), n)
    Pgrid = np.zeros(n)
    Qgrid = np.zeros(n)

    for i, R in enumerate(Rgrid):
        phi = potential.potential([R, 0, 0])

        def fy(y):
            return potential.potential([0, y, 0]) - phi

        def fz(z):
            return potential.potential([0, 0, z]) - phi

        Qgrid[i] = brentq(fy, 1e-6 * R, 10 * R) / R
        Pgrid[i] = brentq(fz, 1e-6 * R, 10 * R) / R

    P_of_R = interp1d(Rgrid, Pgrid, bounds_error=False, fill_value=(Pgrid[0], Pgrid[-1]))
    Q_of_R = interp1d(Rgrid, Qgrid, bounds_error=False, fill_value=(Qgrid[0], Qgrid[-1]))

    return P_of_R, Q_of_R


# Gas mass inside an equipotential radius.
def enclosed_gas_mass(Rmax, rho0, rc, beta, P_of_R, Q_of_R):
    def integrand(R):
        return gas_density(R, rho0, rc, beta) * R**2 * float(P_of_R(R)) * float(Q_of_R(R))

    return 4 * np.pi * quad(integrand, 0, Rmax, epsabs=0, epsrel=1e-5)[0]


# Sample gas particles from the beta-model mass distribution.
def sample_gas_positions(Ngas, Rmax, rc, beta, P_of_R, Q_of_R):
    Rgrid = np.linspace(0, Rmax, 30000)
    Rlookup = np.maximum(Rgrid, Rgrid[1])

    rho = gas_density(Rgrid, 1.0, rc, beta)
    dM = rho * Rgrid**2 * P_of_R(Rlookup) * Q_of_R(Rlookup)

    cdf = np.cumsum(0.5 * (dM[1:] + dM[:-1]) * np.diff(Rgrid))
    cdf = np.insert(cdf, 0, 0.0)
    cdf /= cdf[-1]

    R = interp1d(cdf, Rgrid, bounds_error=False, fill_value=(0, Rmax))(rng.random(Ngas))
    mu = rng.uniform(-1, 1, Ngas)
    phi = rng.uniform(0, 2 * np.pi, Ngas)

    sintheta = np.sqrt(1 - mu**2)
    P = P_of_R(np.maximum(R, Rgrid[1]))
    Q = Q_of_R(np.maximum(R, Rgrid[1]))

    x = R * sintheta * np.cos(phi)
    y = Q * R * sintheta * np.sin(phi)
    z = P * R * mu

    return np.column_stack((x, y, z)), R


# Compute the hydrostatic internal energy profile.
def make_internal_energy(potential, Rmax, rho0, rc, beta):
    Rgrid = np.linspace(0.001 * Rmax, Rmax, 2000)
    rho = gas_density(Rgrid, rho0, rc, beta)
    dphidR = np.zeros(len(Rgrid))

    for i, R in enumerate(Rgrid):
        dphidR[i] = -potential.force([R, 0, 0])[0]

    integrand = rho * dphidR
    integral = np.zeros(len(Rgrid))

    for i in range(len(Rgrid) - 2, -1, -1):
        dR = Rgrid[i + 1] - Rgrid[i]
        integral[i] = integral[i + 1] + 0.5 * (integrand[i + 1] + integrand[i]) * dR

    u = 1.5 * integral / rho
    u = np.maximum(u, 1e-10)

    return interp1d(Rgrid, u, bounds_error=False, fill_value=(u[0], u[-1]))


# Read the sampled dark matter realization.
def read_dm(filename):
    with h5py.File(filename, "r") as file:
        coordinates = file["PartType1/Coordinates"][:]
        velocities = file["PartType1/Velocities"][:]
        masses = file["PartType1/Masses"][:]

    return coordinates, velocities, masses


# Write the complete gas and dark matter cluster.
def write_cluster(filename, gas_pos, gas_vel, gas_mass, gas_u, dm_pos, dm_vel, dm_mass, box_size=20000):
    Ngas = len(gas_mass)
    Ndm = len(dm_mass)

    with h5py.File(filename, "w") as file:
        header = file.create_group("Header")
        numpart = np.array([Ngas, Ndm, 0, 0, 0, 0], dtype=np.uint32)

        header.attrs["NumPart_ThisFile"] = numpart
        header.attrs["NumPart_Total"] = numpart
        header.attrs["NumPart_Total_HighWord"] = np.zeros(6, dtype=np.uint32)
        header.attrs["MassTable"] = np.zeros(6)
        header.attrs["Time"] = 0.0
        header.attrs["Redshift"] = 0.0
        header.attrs["BoxSize"] = box_size
        header.attrs["NumFilesPerSnapshot"] = 1
        header.attrs["Omega0"] = 0.0
        header.attrs["OmegaLambda"] = 0.0
        header.attrs["HubbleParam"] = 1.0
        header.attrs["Flag_Sfr"] = 0
        header.attrs["Flag_Cooling"] = 0
        header.attrs["Flag_StellarAge"] = 0
        header.attrs["Flag_Metals"] = 0
        header.attrs["Flag_Feedback"] = 0
        header.attrs["Flag_DoublePrecision"] = 1
        header.attrs["Flag_IC_Info"] = 0

        gas = file.create_group("PartType0")
        gas.create_dataset("Coordinates", data=gas_pos)
        gas.create_dataset("Velocities", data=gas_vel)
        gas.create_dataset("Masses", data=gas_mass)
        gas.create_dataset("InternalEnergy", data=gas_u)
        gas.create_dataset("ParticleIDs", data=np.arange(1, Ngas + 1, dtype=np.uint64))

        dm = file.create_group("PartType1")
        dm.create_dataset("Coordinates", data=dm_pos)
        dm.create_dataset("Velocities", data=dm_vel)
        dm.create_dataset("Masses", data=dm_mass)
        dm.create_dataset("ParticleIDs", data=np.arange(Ngas + 1, Ngas + Ndm + 1, dtype=np.uint64))


# Add equipotential gas to an AGAMA dark matter halo.
def add_triaxial_gas(dm_file, output, M200, concentration, P, Q, rc_fraction, beta=2 / 3, Ngas=100000):
    R200, rs, potential = make_dm_potential(M200, concentration, P, Q)
    dm_pos, dm_vel, dm_mass = read_dm(dm_file)

    rc = rc_fraction * rs

    Rmin = 0.001 * R200
    Rsample = 2 * R200
    Rpq = 30 * R200
    Ru = 10 * R200

    P_of_R, Q_of_R = make_pq_lookup(potential, Rmin, Rpq)

    Mgas_R200 = fgas / (1 - fgas) * M200
    mass_unit = enclosed_gas_mass(R200, 1.0, rc, beta, P_of_R, Q_of_R)
    rho0 = Mgas_R200 / mass_unit

    Mgas_total = enclosed_gas_mass(Rsample, rho0, rc, beta, P_of_R, Q_of_R)

    gas_pos, Rgas = sample_gas_positions(Ngas, Rsample, rc, beta, P_of_R, Q_of_R)
    gas_vel = np.zeros_like(gas_pos)
    gas_mass = np.full(Ngas, Mgas_total / Ngas)

    u_of_R = make_internal_energy(potential, Ru, rho0, rc, beta)
    gas_u = u_of_R(np.maximum(Rgas, Rmin))

    write_cluster(output, gas_pos, gas_vel, gas_mass, gas_u, dm_pos, dm_vel, dm_mass)


dm_file = "triaxial_dm.hdf5"
output = "triaxial_cluster.hdf5"

M200 = 1.49e14 / MUNIT
concentration = 3.63

P = 0.61
Q = 0.68

rc_fraction = 1 / 5
Ngas = 100000

add_triaxial_gas(dm_file, output, M200, concentration, P, Q, rc_fraction, Ngas=Ngas)