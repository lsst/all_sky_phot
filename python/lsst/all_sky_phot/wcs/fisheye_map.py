import numpy as np
from scipy.interpolate import interp2d, RectBivariateSpline, LinearNDInterpolator
from scipy.spatial import KDTree
from scipy.optimize import minimize


__all__ = ['Fisheye', 'fit_xyshifts']


class Fisheye(object):
    """A wrapper for adding distortions to a rough WCS on an all-sky camera

    To predict the position of a star, the wcs is used to convert to the u,v plane.
    The u,v corrdinates are then shifted by the proper xshift,yshift values to reach the
    final x,y chip coordinates.

    There are of course other, possibly better ways of handling this. Ideally, the SIP
    terms could be used (but I couldn't get them to converge well). Or, one could have nested
    wcs solutions (e.g., one first stage wcs, then an array of wcs solutions for different areas of the chip)
    """

    def __init__(self, wcs, x, y, xshift, yshift):
        """
        Parameters
        ----------
        wcs : astropy wcs object
            The rough WCS of the camera. Note, this assumes that the
            wcs header _says_ RA,Dec but is really a mapping to Az,Alt.
        x : array
            x-positions on the chip where shifts are computed
        y : array
            y-positions on the chip where shifts are computed
        xshift : array
           The amount to shift in x
        yshift : array
           The amount to shift in y.
        """
        self.wcs = wcs
        self.x = x
        self.y = y
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

    def save(self, filename):
        np.savez(filename, wcs=self.wcs, x=self.x, y=self.y,
                 xshift=self.xshift, yshift=self.yshift)

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


def load_fisheye(filename):
    """Load a Fisheye object from a savefile
    """
    data = np.load(filename)
    result = Fisheye(data['wcs'], data['x'], data['y'], data['xshift'], data['yshift'])
    return result


class dist_minimizer(object):
    """Find the x,y shifts that minimizes the median distance between nearest neigbors.

    Note the use of a median here should make it possible to use a refernce catalog where
    not all the stars are in the observed image. However, if most of the reference stars
    are absent I think this will fail.
    """
    def __init__(self, x, y, kdtree):
        self.x = x
        self.y = y
        self.kdtree = kdtree

    def __call__(self, x0):
        xx = self.x + x0[0]
        yy = self.y + x0[1]
        distances, indx = self.kdtree.query(np.array((xx, yy)).T)
        return np.median(distances)


def fit_xyshifts(x, y, alt, az, wcs, max_shift=40., min_points=3, windowsize=200,
                 xrange=[800, 5000], yrange=[0, 3700], num_points=20):
    """Generate a best-fit distortion map.

    Parameters
    ----------
    x : array
        The x-positions of detected objects on the chip
    y : array
        The y-positions of detected objects on the chip
    alt : array
        Altitudes of objects from a reference catalog
    az : array
        Azimuths of objects from a reference catalog
    wcs : astropy.wcs object
        The rough WCS for the chip
    max_shift : float (20.)
        The maximum number of pixels (in x or y) to allow
    min_points : int (3)
        The number of catalog objects that need to be present to attempt a fit.
    windowsize : int (200)
        The square around a gripoint to consider objects
    xrange : list
        The x region on the chip to sample
    yrange : list
        The y region on the chip to sample
    num_points : int (20)
        The number of gridpoints to use in the x and y directions.
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
    npts = xgridpts.ravel()*0.
    for i, (xp, yp) in enumerate(zip(xgridpts.ravel(), ygridpts.ravel())):
        good = np.where((np.abs(catalog_u-xp) < windowsize/2.) & (np.abs(catalog_v-yp) < windowsize/2.))
        if good[0].size < min_points:
            xoffs[i] = 0
            yoffs[i] = 0
            npts[i] = good[0].size
        else:
            fun = dist_minimizer(catalog_u[good], catalog_v[good], observed_kd)
            best_fit = minimize(fun, [0, 0])
            if np.max(np.abs(best_fit.x)) > max_shift:
                xoffs[i] = 0
                yoffs[i] = 0
                npts[i] = good[0].size
            else:
                xoffs[i] = best_fit.x[0]
                yoffs[i] = best_fit.x[1]
                distances[i] = best_fit.fun
                npts[i] = good[0].size

    return xgridpts, ygridpts, xoffs, yoffs, npts, distances


def distortion_mapper(observed_x, observed_y, observed_mjd, catalog_alt, catalog_az, catalog_mjd, wcs,
                      mjd_multiplier=1e6):
    """Try to find a set of distorions that do a good job matching the catalog alt, az positions to
    observed chip positions
    """

    # For each unique mjd, construct a kdtree that can be querried I guess.
    catalog_u, catalog_v = wcs.all_world2pix(catalog_az, catalog_alt, 0)

    # Let's make things 3 dimensional, so we can use the kdtree to do the mjd sorting for us
    observed_kd = KDTree(np.array([observed_x, observed_y, observed_mjd*mjd_multiplier]).T)

    # Make a grid of u,v reference points. Assign each catalog point to an offset point.
    ugrid = np.linspace(np.min(urange), np.max(urange), num_points)
    vgrid = np.linspace(np.min(vrange), np.max(vrange), num_points)

    ugrid, vgrid = np.meshgrid(ugrid, vgrid)
    ugrid = ugrid.ravel()
    vgrid = vgrid.ravel()

    ref_kd = KDTree(np.array(ugrid, vgrid).T)

    # Find the reference index for each catalog observation
    distances, indx = ref_kd.query(np.array(catalog_u, catalog_v).T)

    # Now I can make a function that make a grid of x offsets and y offsets, applies them to the
    # catalog u,v corrds, and finds the median distance to observed points.

    pass
