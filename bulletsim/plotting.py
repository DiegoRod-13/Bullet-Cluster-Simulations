import numpy as np

from photutils.detection import find_peaks
from scipy.ndimage import gaussian_filter

from .snapshot import read_dm, xyz


def dm_map(snapshot, view=(-900, 900, -900, 900), bins=100, sigma=1):
    """Create a smoothed projected dark matter mass map."""

    xmin, xmax, ymin, ymax = view
    pos, mass, _ = read_dm(snapshot)

    H, xedges, yedges = np.histogram2d(
        pos[:, 0], pos[:, 1], bins=bins,
        range=[[xmin, xmax], [ymin, ymax]], weights=mass
    )

    H = gaussian_filter(H.T, sigma)

    xcent = 0.5 * (xedges[1:] + xedges[:-1])
    ycent = 0.5 * (yedges[1:] + yedges[:-1])

    return H, xcent, ycent


def plot_dm_contours(ax, snapshot, view=(-900, 900, -900, 900), bins=100,
                     sigma=1, percentiles=(50, 70, 80, 95, 99.5)):
    """Plot projected dark matter contours."""

    H, xcent, ycent = dm_map(snapshot, view, bins, sigma)
    positive = H[H > 0]

    if len(positive) == 0:
        return H, xcent, ycent

    levels = np.unique(np.percentile(positive, percentiles))

    if len(levels) > 1:
        ax.contour(xcent, ycent, H, levels=levels, colors="white", linewidths=1)

    return H, xcent, ycent


def dm_peaks(snapshot, center=[0, 0, 0], bins=100, box=1800, sigma=3, threshold=0.3):
    """Find the two strongest projected dark matter peaks."""

    half = box / 2
    pos, mass, _ = read_dm(snapshot)
    x, y, _ = xyz(pos, center)

    H, xedges, yedges = np.histogram2d(
        x, y, bins=bins,
        range=[[-half, half], [-half, half]], weights=mass
    )

    H = gaussian_filter(H.T, sigma)

    xcent = 0.5 * (xedges[1:] + xedges[:-1])
    ycent = 0.5 * (yedges[1:] + yedges[:-1])

    peaks = find_peaks(H, threshold=threshold * H.max())
    peaks.sort("peak_value")
    peaks.reverse()

    if len(peaks) == 0:
        return None, None, None

    peak1 = peaks[0]
    center1 = [xcent[peak1["x_peak"]], ycent[peak1["y_peak"]]]

    if len(peaks) == 1:
        return center1, None, None

    peak2 = peaks[1]
    center2 = [xcent[peak2["x_peak"]], ycent[peak2["y_peak"]]]

    separation = np.sqrt(
        (center1[0] - center2[0])**2
        + (center1[1] - center2[1])**2
    )

    return center1, center2, separation


def plot_dm_peaks(ax, snapshot, **kwargs):
    """Plot the two strongest dark matter peaks."""

    center1, center2, separation = dm_peaks(snapshot, **kwargs)

    for center in [center1, center2]:
        if center is not None:
            ax.scatter(center[0], center[1], marker="+", s=80,
                       color="cyan", linewidths=2)

    return center1, center2, separation