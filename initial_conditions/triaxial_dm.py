import agama
import h5py
import numpy as np

from scipy.integrate import quad


MUNIT = 1e10

agama.setUnits(length=1, velocity=1, mass=MUNIT)

G = 4.30091727003628e-6 * MUNIT

z = 0.296
H0 = 70.0
OmegaM = 0.3
OmegaL = 0.7


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

    integral = quad(integrand, 1e-8 * radius, radius, epsabs=0, epsrel=1e-6)[0]
    return 4 * np.pi * P * Q * integral


def density_norm_for_m200(M200, R200, rs, P, Q):
    density = agama.Density(type="Spheroid", axisRatioY=Q, axisRatioZ=P, scaleRadius=rs, densityNorm=1.0, alpha=1, beta=3, gamma=1, outerCutoffRadius=2 * R200, cutoffStrength=8)
    mass_unit = enclosed_mass_ellipsoid(density, R200, P, Q)
    return M200 / mass_unit


def write_gadget_hdf5(filename, result, box_size=10000.0):
    xv, masses = result

    coordinates = np.asarray(xv[:, :3])
    velocities = np.asarray(xv[:, 3:6])
    masses = np.asarray(masses)
    particle_ids = np.arange(1, len(masses) + 1)

    with h5py.File(filename, "w") as file:
        header = file.create_group("Header")

        numpart = np.array([0, len(masses), 0, 0, 0, 0])

        header.attrs["NumPart_ThisFile"] = numpart
        header.attrs["NumPart_Total"] = numpart
        header.attrs["NumPart_Total_HighWord"] = np.zeros(6)
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

        dm = file.create_group("PartType1")

        dm.create_dataset("Coordinates", data=coordinates)
        dm.create_dataset("Velocities", data=velocities)
        dm.create_dataset("Masses", data=masses)
        dm.create_dataset("ParticleIDs", data=particle_ids)


def make_triaxial_dm(name, M200, concentration, P, Q, n_orbits=10000, nbody=100000):
    R200 = r200_from_m200(M200, z)
    rs = R200 / concentration
    Rmax = 2 * R200

    density_norm = density_norm_for_m200(M200, R200, rs, P, Q)
    density = agama.Density(type="Spheroid", axisRatioY=Q, axisRatioZ=P, scaleRadius=rs, densityNorm=density_norm, alpha=1, beta=3, gamma=1, outerCutoffRadius=Rmax, cutoffStrength=8)
    potential = agama.Potential(type="Multipole", lmax=8, mmax=6, density=density)

    density_target = agama.Target(type="DensityClassicLinear", gridr=agama.nonuniformGrid(20, 0.1 * R200, R200), axisratioy=Q, axisratioz=P, stripsPerPane=2)
    kinematic_target = agama.Target(type="KinemShell", gridR=agama.nonuniformGrid(15, 0.1 * R200, Rmax), degree=1)

    initial_conditions, _ = density.sample(n_orbits, potential=potential)

    density_data, kinematic_data, trajectories = agama.orbit(potential=potential, ic=initial_conditions, time=50 * potential.Tcirc(initial_conditions), dtype=object, targets=[density_target, kinematic_target])

    mass_data = np.ones((n_orbits, 1))
    density_rhs = density_target(density)

    ngrid = len(kinematic_target) // 2
    kinematic_matrix = 2 * kinematic_data[:, :ngrid] - kinematic_data[:, ngrid:]
    kinematic_rhs = np.zeros(ngrid)

    mass = enclosed_mass_ellipsoid(density, Rmax, P, Q)

    weights = agama.solveOpt(matrix=(mass_data.T, density_data.T, kinematic_matrix.T), rhs=([mass], density_rhs, kinematic_rhs), rpenl=([np.inf], np.ones_like(density_rhs), np.ones_like(kinematic_rhs) * 1e-3), xpenq=np.ones(n_orbits) * 0.03)

    xv, masses = agama.sampleOrbitLibrary(nbody, trajectories, weights)

    R = np.sqrt(xv[:, 0]**2 + (xv[:, 1] / Q)**2 + (xv[:, 2] / P)**2)
    masses *= M200 / masses[R <= R200].sum()

    write_gadget_hdf5(f"{name}.hdf5", (xv, masses))


name = "triaxial_dm"

M200 = 1.49e14 / MUNIT
concentration = 3.63

P = 0.61
Q = 0.68

n_orbits = 15000
nbody = 80000

make_triaxial_dm(name, M200, concentration, P, Q, n_orbits=n_orbits, nbody=nbody)