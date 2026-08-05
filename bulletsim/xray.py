import numpy as np
import pyxsim
import soxs
import yt

from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits


cosmo = FlatLambdaCDM(H0=70, Om0=0.3)


def mock_chandra_map(snapshot, emin=0.5, emax=2, redshift=0.296, metallicity=0, box=1800, npix=400, proj_angle=0, view_angle=0):
    """Create a synthetic Chandra flux map."""

    ds = yt.load(snapshot)
    XH = 0.76

    def emission_measure(field, data):
        rho = data[("gas", "density")]
        mass = data[("gas", "mass")]
        mp = data.ds.quan(1.6726192369e-24, "g")

        nH = XH * rho / mp
        ne = (XH + 0.5 * (1 - XH)) * rho / mp
        volume = mass / rho

        return ne * nH * volume

    ds.add_field(("gas", "emission_measure"), function=emission_measure, sampling_type="particle", units="cm**-3")

    model = pyxsim.CIESourceModel("apec", emin, emax, 1000, Zmet=metallicity)

    exp_time = (558, "ks")
    area = (2000, "cm**2")
    half = box / 2

    kpc = cosmo.kpc_proper_per_arcmin(redshift).value
    fov = box / kpc

    inst_name = f"bullet_{int(box)}kpc_{2*npix}px"

    lower = ds.domain_center - ds.arr([half, half, half], "kpc")
    upper = ds.domain_center + ds.arr([half, half, half], "kpc")
    source = ds.box(lower, upper)

    try:
        soxs.make_simple_instrument("chandra_acisi_cy22", inst_name, fov, npix)
    except KeyError:
        pass

    theta = np.deg2rad(proj_angle)
    alpha = np.deg2rad(view_angle)

    normal = [np.sin(theta), 0, np.cos(theta)]
    north = [-np.sin(alpha), np.cos(alpha), 0]

    pyxsim.make_photons("photons", source, redshift, area, exp_time, model)

    pyxsim.project_photons("photons", "events", normal, [104.65, -55.95], absorb_model="tbabs", nH=0.05, flat_sky=True, kernel="gaussian", north_vector=north)

    soxs.instrument_simulator("events.h5", "mock_evt.fits", exp_time, inst_name, [104.65, -55.95], overwrite=True, instr_bkgnd=True, ptsrc_bkgnd=False)

    soxs.make_exposure_map("mock_evt.fits", "expmap.fits", energy=2, overwrite=True)

    soxs.write_image("mock_evt.fits", "flux.fits", emin=emin, emax=emax, expmap_file="expmap.fits", overwrite=True)

    image = fits.getdata("flux.fits")
    image /= 558 * 1000

    return image