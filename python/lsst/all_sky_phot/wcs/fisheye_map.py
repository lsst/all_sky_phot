import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import KDTree
from scipy.optimize import minimize
import sys
import matplotlib.pylab as plt

__all__ = ['Fisheye', 'distortion_mapper', 'load_fisheye', 'distortion_mapper_looper']


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

    def all_world2pix(self, az, alt, ref):
        """Convert az,alt to chip x,y

        Parameters:
        -----------
        alt : array
            Altitude of points (degrees)
        az : array
            Azimuth of points
        ref : int
            Reference pixel for the wcs object (0 is a good choice)

        Returns
        -------
        x : float
            chip position (pixels)
        y : float
            chip position (pixels)
        """
        u, v = self.wcs.all_world2pix(az, alt, ref)
        x = u + self.xinterp(u, v)
        y = v + self.yinterp(u, v)
        return x, y

        pass

    def all_pix2world(self, x, y, ref):
        """Convert a chip x,y to altitude, azimuth

        Parameters
        ----------
        x : array
            x pixel corrdinate
        y : array
            y pixel coordinate
        ref : int
            Reference pixel for the wcs object (0 is a good choice)

        Returns
        -------
        az : array
            The azimuth (degrees)
        alt : array
            Altitude (degrees)
        """
        u = x + self.reverse_xinterp(x, y)
        v = y + self.reverse_yinterp(x, y)
        az, alt = self.wcs.all_pix2world(u, v, ref)

        return az, alt


def load_fisheye(filename):
    """Load a Fisheye object from a savefile
    """
    data = np.load(filename)
    result = Fisheye(data['wcs'][()], data['x'], data['y'], data['xshift'], data['yshift'])
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


class dist_mapper_minimizer(object):
    """Wrapper that makes it easy to solve for wcs shifts with standard scipy minimizers.
    """
    def __init__(self, catalog_u, catalog_v, mjd, kdtree):
        """
        Parameters
        ----------
        catalog_u : array
            Catalog positions that have been projected to the u,v plane
        catalog_v : array
            Catolog v positions
        mjd : array
            The MJD for each observed point
        kdtree : scipy kdtree
            A KD tree built from observed x,y,mjd points.
        """
        self.u = catalog_u
        self.v = catalog_v
        self.mjd = mjd
        self.kdtree = kdtree

    def __call__(self, x0):
        """
        Parameters
        ----------
        x0 : 2-element array

        Returns
        -------
        Median distance between catalog points and their nearest observed neighbor after
        shifts have been applied.
        """
        xshifts = x0[0]
        yshifts = x0[1]

        final_x = self.u + xshifts
        final_y = self.v + yshifts

        distances, indx = self.kdtree.query(np.array([final_x, final_y, self.mjd]).T)

        result = np.median(distances)
        return result


def distortion_mapper(observed_x, observed_y, observed_mjd, catalog_alt, catalog_az, catalog_mjd, wcs,
                      mjd_multiplier=1e4, u_center=2000, v_center=2000, window=100, pad=20, diaog=False):
    """Try to find a set of distortions that do a good job matching the catalog alt, az positions to
    observed chip positions
    """

    # Generate the u,v positions of catalog objects
    catalog_u, catalog_v = wcs.all_world2pix(catalog_az, catalog_alt, 0)

    # Crop down to the region of the chip we are interested in
    rad = ((catalog_u-u_center)**2+(catalog_v-v_center)**2)**0.5
    good = np.where(rad < window)

    catalog_u = catalog_u[good]
    catalog_v = catalog_v[good]
    catalog_mjd = catalog_mjd[good]

    rad = ((observed_x-u_center)**2+(observed_y-v_center)**2)**0.5
    good = np.where(rad < window+pad)
    observed_x = observed_x[good]
    observed_y = observed_y[good]
    observed_mjd = observed_mjd[good]

    # If we have nothing to match, return None
    if (catalog_u.size == 0) | (observed_x.size == 0):
        return None

    # Let's make things 3 dimensional, so we can use the kdtree to do the mjd sorting for us
    observed_kd = KDTree(np.array([observed_x, observed_y, observed_mjd*mjd_multiplier]).T)

    fun = dist_mapper_minimizer(catalog_u, catalog_v, catalog_mjd*mjd_multiplier,
                                observed_kd)
    x0 = np.array([0., 0.])
    result = minimize(fun, x0, method='Powell')
    # Because I can
    result.ncat = catalog_mjd.size
    result.nobs = observed_mjd.size


    #if (u_center > 2200) & (u_center < 2300) & (v_center > 500) & (v_center < 600) & (np.abs(result.x[0]) > 10) & (np.abs(result.x[1]) > 10):
    #import pdb ; pdb.set_trace()

    if diaog:
        fig, ax = plt.subplots()
        ax.plot(catalog_u+result.x[0], catalog_v+result.x[1], 'go', alpha=.5)
        ax.plot(observed_x, observed_y, 'ro', alpha=.5)
        return result, ax
    else:
        return result


def distortion_mapper_looper(observed_x, observed_y, observed_mjd, catalog_alt, catalog_az, catalog_mjd, wcs,
                             xmax=5796, ymax=3870, nx=20, ny=20, mjd_multiplier=1e4,
                             window=100, pad=20, verbose=True):
    """Given observed and expected stellar catalogs and a rough WCS, fit an additional distortion map.

    Parameters
    ----------
    observed_x : array
        The observed x-positions of stars
    observed_y : array
        The observed y-position of stars
    observed_mjd : array
        The MJD for each observation (so multiple frames can be fit simultaneously)
    catalog_alt : array
        Altitudes of stars expected to be in the frame(s) (degrees)
    catalog_az : array
        Azimuths of stars expected to be in the frame(s) (degrees)
    catalog_mjd : array
        The MJDs for the catalogs
    wcs : wcs object
        A rough WCS that transforms catalog alt,az to chip x,y
    xmax : int (5796)
        The maximum x-position to try and fit on the chip
    ymax : int (3870)
        The maximum y-position to try and fit
    nx : int (20)
        The number of gridpoints to use in the x-dimension
    ny : int (20)
        The number of gridpoints to use in the y-dimension
    mjd_multiplier : float (1e4)
        MJD values are multiplied by this to make KD-tree generation. Should be order-of-magnitude
        larger than xmax and ymax.
    window : int (100)
        The window size to select around each gridpoint (pixels).
    pad : int (20)
        Pad on the window in case stars we want fall outside (pixels)
    verbose : bool (True)
        Print out a progress bar
    """
    xgrid = np.linspace(0, xmax, nx)
    ygrid = np.linspace(0, ymax, ny)

    xp = []
    yp = []
    xshifts = []
    yshifts = []
    distances = []
    npts = []
    i = 0
    imax = float(xgrid.size*ygrid.size)
    for u_center in xgrid.ravel():
        for v_center in ygrid.ravel():
            xp.append(u_center)
            yp.append(v_center)
            fit_result = distortion_mapper(observed_x, observed_y, observed_mjd, catalog_alt,
                                           catalog_az, catalog_mjd, wcs, window=window,
                                           u_center=u_center, v_center=v_center)
            if fit_result is None:
                xshifts.append(np.nan)
                yshifts.append(np.nan)
                distances.append(np.nan)
                npts.append(np.nan)
            else:
                xshifts.append(fit_result.x[0])
                yshifts.append(fit_result.x[1])
                distances.append(fit_result.fun)
                npts.append(fit_result.nobs)
            if verbose:
                i += 1
                progress = i/imax*100
                text = "\rprogress = %.2f%%" % progress
                sys.stdout.write(text)
                sys.stdout.flush()
    yp = np.array(yp)
    xp = np.array(xp)
    xshifts = np.array(xshifts)
    yshifts = np.array(yshifts)
    distances = np.array(distances)
    npts = np.array(npts)

    result = {'yp': yp, 'xp': xp, 'xshifts': xshifts, 'yshifts': yshifts,
              'distances': distances, 'npts': npts}

    good = np.where(~(np.isnan(xshifts)) & ~(np.isnan(yshifts)))
    wcs_w_shift = Fisheye(wcs, xp[good], yp[good], xshifts[good], yshifts[good])
    return wcs_w_shift, result


