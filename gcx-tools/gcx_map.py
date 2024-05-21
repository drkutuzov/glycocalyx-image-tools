import numpy as np
from scipy.special import erf
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

def smeared_gauss(x, y, s, d):
    """
    2D rotationally-symmetric Gaussian convolved with a line segment of width 'd' along the x-axis
    
    Parameters
    ----------
    x : 2D array_like
    y : 2D array_like
    s : float
        STD of the glycocalyx peak
    d : float
        width of the line segment

    Returns
    -------
    z: 2D array_like
    """
    x1, x2 = (x - d/2)/np.sqrt(2)/s, (x + d/2)/np.sqrt(2)/s
    return 0.5*(erf(x2) - erf(x1))*np.exp(-y**2/2/s**2)


def smeared_gauss_rotated(x, y, s, d, theta):
    """
    'smeared_gauss' rotated by theta (in radians)
    """
    x_rot, y_rot = np.cos(theta)*x - np.sin(theta)*y, np.sin(theta)*x + np.cos(theta)*y
    return smeared_gauss(x_rot, y_rot, s, d)


def gcx_segment(x, y, A, s, d, xo, yo, theta):
    """ 
    Plots a segment of the glycocalyx 

    Parameters
    ----------
    x : 2D array_like
    y : 2D array_like
    A : float 
        Maximum value of the glycocalyx fluorescence intensity profile
    s : float
        STD of the glycocalyx peak
    d : float
        width of the segment along which line-profiles of intensity are averaged
    xo : float
        x-coordinate of the glycocalyx peak (maximum intensity)
    yo : float 
        y-coordinate of the glycocalyx peak (maximum intensity)
    theta : float
        Angle between the vessel wall segment and x-axis, in radians

    Returns
    -------
    z : 2D array_like

    Notes
    -----
    Use 'get_angle' to calculate the angle

    """
    return A*smeared_gauss_rotated(x - xo, y - yo, s, d, theta)


def get_angle(point1, point2):
    """ 
    Angle (in radians) perpendicular to a line segments with two ends with coordinates 'point1' and 'point2'
    where point1 = (x1, y1), point2 = (x2, y2)
    """
    x1, y1 = point1
    x2, y2 = point2
    return np.pi/2 - np.arctan((y2 -y1)/(x2 - x1)) if x1 != x2 else 0


def plot_line_segments(x, y, vals, colormap, linewidth=3, ax=None, xlim=None, ylim=None, xlabel=None, ylabel=None, cbar_label=None):
    """ 
    Plots line segments of different colors

    Parameters
    ----------
    x : 1D array_like
    y : 1D array_like
    vals: 1D array_like
          values corresponding to each (x, y) point, e.g. brightness of the glycocalyx
    colormap: str or a mapping
    linewidth: float
               line width of the segment
    cbar_label
    """
    if ax is None:
        fig, ax = plt.subplots()
    
    vals_mean = 0.5*(vals[:-1] + vals[1:])
    vals_norm = plt.Normalize(vals_mean.min(), vals_mean.max())
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    lc = LineCollection(segments, cmap=colormap, norm=vals_norm)
    lc.set_array(vals_mean)
    lc.set_linewidth(linewidth)
    line = ax.add_collection(lc)
    cbar = plt.colorbar(line, ax=ax)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    cbar.set_label(cbar_label)