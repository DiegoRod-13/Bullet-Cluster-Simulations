import h5py
import numpy as np


def rotation_matrix_x(angle):
    angle = np.deg2rad(angle)
    return np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]])


def rotation_matrix_y(angle):
    angle = np.deg2rad(angle)
    return np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])


def rotation_matrix_z(angle):
    angle = np.deg2rad(angle)
    return np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])


def rotation_matrix(angles):
    x_angle, y_angle, z_angle = angles
    return rotation_matrix_z(z_angle) @ rotation_matrix_y(y_angle) @ rotation_matrix_x(x_angle)


# The center of each isolated cluster must be known before combining.
def combine_clusters(main_file, bullet_file, output, main_source_center, bullet_source_center, separation, impact_parameter, velocity, main_angles=(0, 0, 0), bullet_angles=(0, 0, 0), box_size=20000):
    main_source_center = np.asarray(main_source_center)
    bullet_source_center = np.asarray(bullet_source_center)

    main_center = np.array([0.0, 0.0, 0.0])
    bullet_center = np.array([-separation, impact_parameter, 0.0])

    main_rotation = rotation_matrix(main_angles)
    bullet_rotation = rotation_matrix(bullet_angles)

    with h5py.File(main_file, "r") as main, h5py.File(bullet_file, "r") as bullet, h5py.File(output, "w") as out:
        main.copy("Header", out)

        numpart = main["Header"].attrs["NumPart_ThisFile"] + bullet["Header"].attrs["NumPart_ThisFile"]
        out["Header"].attrs["NumPart_ThisFile"] = numpart
        out["Header"].attrs["NumPart_Total"] = numpart
        out["Header"].attrs["NumPart_Total_HighWord"] = np.zeros(6, dtype=np.uint32)
        out["Header"].attrs["BoxSize"] = box_size

        particle_id = 1
        particle_types = sorted(set(main.keys()) & set(bullet.keys()))
        particle_types = [ptype for ptype in particle_types if ptype.startswith("PartType")]

        for ptype in particle_types:
            group = out.create_group(ptype)

            main_pos = main[ptype]["Coordinates"][:] - main_source_center
            bullet_pos = bullet[ptype]["Coordinates"][:] - bullet_source_center
            main_vel = main[ptype]["Velocities"][:]
            bullet_vel = bullet[ptype]["Velocities"][:]

            main_pos = main_pos @ main_rotation.T
            bullet_pos = bullet_pos @ bullet_rotation.T
            main_vel = main_vel @ main_rotation.T
            bullet_vel = bullet_vel @ bullet_rotation.T

            main_pos += main_center
            bullet_pos += bullet_center
            bullet_vel[:, 0] += velocity

            coordinates = np.vstack((main_pos, bullet_pos))
            velocities = np.vstack((main_vel, bullet_vel))

            group.create_dataset("Coordinates", data=coordinates)
            group.create_dataset("Velocities", data=velocities)

            fields = sorted(set(main[ptype].keys()) & set(bullet[ptype].keys()))
            fields = [field for field in fields if field not in ["Coordinates", "Velocities", "ParticleIDs"]]

            for field in fields:
                data = np.concatenate((main[ptype][field][:], bullet[ptype][field][:]))
                group.create_dataset(field, data=data)

            ids = np.arange(particle_id, particle_id + len(coordinates), dtype=np.uint64)
            group.create_dataset("ParticleIDs", data=ids)
            particle_id += len(coordinates)


main_file = "main_cluster.hdf5"
bullet_file = "bullet_cluster.hdf5"
output = "collision.hdf5"

# Cluster Generator files are commonly centered in the box.
# AGAMA files are commonly centered at the origin.
main_source_center = [10000, 10000, 10000]
bullet_source_center = [0, 0, 0]

separation = 7000
impact_parameter = 300
velocity = 1000

main_angles = (0, 0, 0)
bullet_angles = (0, 0, 0)

combine_clusters(main_file, bullet_file, output, main_source_center, bullet_source_center, separation, impact_parameter, velocity, main_angles, bullet_angles)