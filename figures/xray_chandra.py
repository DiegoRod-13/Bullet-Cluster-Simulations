import matplotlib.pyplot as plt
import numpy as np

from astropy.convolution import Gaussian2DKernel, convolve
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.visualization import AsinhStretch, PercentileInterval
from astropy.wcs import WCS
from reproject import reproject_interp


xray_file = "data/chandra_flux.fits"
lensing_file = "data/lensing_kappa.fits"
output = "xray_chandra.png"

# Cluster center selected in DS9.
x_center = 1855
y_center = 2152

size = (800, 800)
pixel_scale = 2.16

xray_hdu = fits.open(xray_file)[0]
xray_data = xray_hdu.data
xray_header = xray_hdu.header.copy()
xray_wcs = WCS(xray_header)

cutout = Cutout2D(xray_data, position=(x_center, y_center), size=size, wcs=xray_wcs)
xray_crop = cutout.data
crop_wcs = cutout.wcs
crop_header = crop_wcs.to_header()

with fits.open(lensing_file) as hdul:
    lensing_data = hdul[0].data
    lensing_header = hdul[0].header.copy()

lensing_reprojected, footprint = reproject_interp((lensing_data, lensing_header), crop_header, shape_out=xray_crop.shape)

kernel = Gaussian2DKernel(x_stddev=2)
lensing_smooth = convolve(lensing_reprojected, kernel, boundary="extend", preserve_nan=True)

valid = lensing_smooth[np.isfinite(lensing_smooth)]
valid = valid[valid > 0]
levels = np.percentile(valid, [50, 70, 80, 90, 95])

transform = AsinhStretch(0.04) + PercentileInterval(99.5)

half_x = 0.5 * xray_crop.shape[1] * pixel_scale
half_y = 0.5 * xray_crop.shape[0] * pixel_scale
extent = [-half_x, half_x, -half_y, half_y]

fig, ax = plt.subplots(figsize=(7, 7))

ax.imshow(transform(xray_crop), origin="lower", cmap="inferno", extent=extent)
ax.contour(lensing_smooth, levels=levels, colors="white", linewidths=1, origin="lower", extent=extent)

ax.set(xlabel="x [kpc]", ylabel="y [kpc]", xlim=(-half_x, half_x), ylim=(-half_y, half_y), aspect="equal")

fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
plt.show()s