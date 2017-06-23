import numpy as np
import healpy as hp
from scipy.interpolate import interp2d, RectBivariateSpline, LinearNDInterpolator
from scipy.spatial import KDTree
from scipy.optimize import minimize


__all__ = ['Fisheye', 'fit_xyshifts']


class Fisheye(object):
    """A wrapper for adding distortions to a rough WCS on an all-sky camera"""

    def __init__(self, wcs, x, y, xshift, yshift):
        """
        Parameters
        ----------

        The transformation is assumed to be 

        wcs : astropy wcs object
            The rough WCS of the camera. Note, this assumes that the 
            wcs _says_ RA,Dec but is really a mapping to Az,Alt.
        xshift : np.array
           The 
        """
        self.wcs = wcs
        # For a point that lands on x,y shift x by xshift and y by yshift
        self.xshift = xshift
        self.yshift = yshift
                
        # If I bothered making things into grids, I could make this a RectBivariateSpline
        self.xinterp = LinearNDInterpolator(np.array([x, y]).T, xshift)
        self.yinterp = LinearNDInterpolator(np.array([x, y]).T, yshift)

        reverse_x = x + xshift
        reverse_y = y + yshift

        self.reverse_xinterp = LinearNDInterpolator(np.array([reverse_x, reverse_y]).T, -xshift)
        self.reverse_yinterp = LinearNDInterpolator(np.array([reverse_x, reverse_y]).T, -yshift)
       
    def all_world2pix(self, az, alt):
        """
        Parameters:
        -----------
        alt : array
            Altitude of points (degrees)
        az : array
            Azimuth of points
        """
        u, v = self.wcs.all_world2pix(az, alt, 0)
        x = self.xinterp(u, v)
        y = self.yinterp(u, v)
        return x, y

        pass

    def all_pix2world(self, x, y):
        """
        Parameters
        ----------
        x : array
           x pixel corrdinate
        y : array
           y pixel coordinate

        Returns
        -------
        az : array
            The azimuth (degrees)
        alt : array
            Altitude (degrees)
        """
        u = self.reverse_xinterp(x, y)
        v = self.reverse_yinterp(x, y)
        az, alt = self.wcs.all_pix2world(u, v, 0)

        return alt, az


class dist_minimizer(object):
    def __init__(self, x, y, kdtree):
        self.x = x
        self.y = y
        self.kdtree = kdtree

    def __call__(self, x0):
        xx = self.x + x0[0]
        yy = self.y + x0[1]
        distances, indx = self.kdtree.query(np.array((xx,yy)).T)
        return np.median(distances)


def fit_xyshifts(x, y, alt, az, wcs, max_shift=20, min_points=3, windowsize=200,
                 xrange=[800, 5000], yrange=[0, 3700], num_points=20):
    """Generate a distortion map
    """
    observed_kd = KDTree(np.array([x, y]).T)

    catalog_u, catalog_v = wcs.all_world2pix(az, alt, 0)

    # Let's set up a grid of x,y point where we want to sample.
    xgridpts = np.linspace(np.min(xrange), np.max(xrange), num_points)
    ygridpts = np.linspace(np.min(yrange), np.max(yrange), num_points)

    xgridpts, ygridpts = np.meshgrid(xgridpts, ygridpts)
    xoffs = xgridpts.ravel()*0.
    yoffs = xgridpts.ravel()*0.
    distances = xgridpts.ravel()*0.
    for i, (xp, yp) in enumerate(zip(xgridpts.ravel(), ygridpts.ravel())):
        good = np.where((np.abs(catalog_u-xp) < windowsize/2.) & (np.abs(catalog_v-yp) < windowsize/2.))
        if good[0].size < min_points:
            xoffs[i] = 0
            yoffs[i] = 0
        else:
            fun = dist_minimizer(catalog_u[good], catalog_v[good], observed_kd)
            best_fit = minimize(fun, [0, 0])
            if np.max(np.abs(best_fit.x)) > max_shift:
                xoffs[i] = 0
                yoffs[i] = 0
            else:
                xoffs[i] = best_fit.x[0]
                yoffs[i] = best_fit.x[1]
                distances[i] = best_fit.fun

    return xgridpts, ygridpts, xoffs, yoffs, distances

