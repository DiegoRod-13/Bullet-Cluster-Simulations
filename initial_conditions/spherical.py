from cluster_generator import ClusterModel
from cluster_generator.radial_profiles import find_overdensity_radius, nfw_scale_density, nfw_density_profile, beta_model_profile, rescale_profile_by_mass


def make_spherical_cluster(name, M200, concentration, fgas, rc_fraction, Ndm, Ngas, z=0.296, beta=2 / 3, box_size=20000):
    R200 = find_overdensity_radius(M200, 200, z=z)
    rs = R200 / concentration

    rhos = nfw_scale_density(concentration, z=z, delta=200)
    rhos *= 1 - fgas

    rho_dm = nfw_density_profile(rhos, rs)

    rc = rc_fraction * rs
    rho_gas = beta_model_profile(1, rc, beta)
    rho_gas = rescale_profile_by_mass(rho_gas, fgas * M200, R200)

    rho_total = rho_dm + rho_gas

    model = ClusterModel.from_dens_and_tden(rmin=1e-3, rmax=5 * R200, density=rho_gas, total_density=rho_total, num_points=20000)

    gas = model.generate_gas_particles(num_particles=Ngas, r_max=R200, compute_potential=True)
    dm = model.generate_dm_particles(num_particles=Ndm, r_max=R200, compute_potential=True)

    center = [box_size / 2, box_size / 2, box_size / 2]

    gas.add_offsets(center, [0, 0, 0])
    dm.add_offsets(center, [0, 0, 0])

    particles = gas + dm
    particles.write_to_gadget_file(f"{name}.hdf5", box_size=box_size, overwrite=True)


name = "spherical_cluster"

M200 = 1.49e14
concentration = 3.63

fgas = 0.13
rc_fraction = 1 / 5

Ndm = 80000
Ngas = 100000

make_spherical_cluster(name, M200, concentration, fgas, rc_fraction, Ndm, Ngas)