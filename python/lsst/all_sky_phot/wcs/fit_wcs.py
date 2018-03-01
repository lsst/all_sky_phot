import numpy as np
from astropy.wcs import Sip
from astropy import wcs
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
import astropy.units as u
from astropy.time import Time

__all__ = ['wcs_azp', 'wcs_zea', 'wcs_refine', 'wcs_refine_zea']


class wcs_azp(object):
    """
    Helper class so WCS parameters can be varied to find the best fit.

    The class includes methods for mapping a vector to WCS parameters and back. Thus,
    things like scipy minimize can be used to vary the WCS values to find the best fit.
    """

    def __init__(self, x, y, alt, az, a_order=2, b_order=2, crpix1=0, crpix2=0, nparams=10):
        """
        Parameters
        ----------
        x : array (float)
            x-positions on the chip. x,y, alt, and az should all be the same length.
        y : array (float)
            y positions on the chip
        alt : array (float)
            Altitudes of the stars (degrees)
        az : array (float)
            Azimuths of the stars (degrees)
        a_order : int (2)
            Order of the SIP distortion in one dimension
        b_order : int (2)
            Order of the SIP distortion in the other dimension
        crpix1 : float (0.)
            The WCS crpix1
        crpix2 : float (0.)
            The WCS crpix2
        nparams : int (10)
            The number of free parameters in the WCS

        """
        # Check that we have the stars matched already
        if len(np.unique([len(x), len(y), len(alt), len(az)])) > 1:
            raise ValueError('x, y, alt, az must all have the same length.')

        self.az = az
        self.alt = alt
        self.x = x
        self.y = y

        # The wcs object we'll be using
        self.w = wcs.WCS(naxis=2)
        self.w.wcs.crpix = [crpix1, crpix2]
        # Fix the reference pixel to zenith
        self.w.wcs.crval = [0, 90]
        #self.w.wcs.ctype = ["RA---ZEA", "DEC--ZEA"]
        self.w.wcs.ctype = ["RA---AZP-SIP", "DEC--AZP-SIP"]

        self.world_coords = np.vstack((az, alt))
        self.pix_coords = np.vstack((x, y))
        # Make a sip object with all zero values
        self.a_order = a_order
        self.b_order = b_order
        n_a = int((a_order + 1.)**2)
        n_b = int((b_order + 1)**2)

        self.a_ind = np.arange(n_a) + nparams
        self.b_ind = np.arange(n_b) + self.a_ind.max() + 1

        self.sip_zeros_a = np.zeros((a_order+1, a_order + 1))
        self.sip_zeros_b = np.zeros((b_order+1, b_order + 1))
        self.w.sip = None

    def set_wcs(self, x0):
        """
        Values in a single vector x0 are mapped to WCS parameters.

        x0 = [0:crpix1, 1:crpix2, 2:cdelt1, 3:cdelt2, 4:pc, 5:pc, 6:pc, 7:pc, 8:mu, 9:gamma, sip... ]
        """
        # Referece Pixel
        self.w.wcs.crpix = [x0[0], x0[1]]

        # Set the cdelt values
        self.w.wcs.cdelt = [x0[2], x0[3]]

        # Set the pc matrix
        self.w.wcs.pc = x0[4:8].reshape((2, 2))

        # Set mu and gamma
        self.w.wcs.set_pv([(2, 1, x0[8]), (2, 2, x0[9])])

        # Make a new SIP
        if np.size(x0) > 10:
            a = x0[self.a_ind].reshape((self.a_order + 1, self.a_order + 1))
            b = x0[self.b_ind].reshape((self.b_order + 1, self.b_order + 1))
            self.w.sip = Sip(a, b, self.sip_zeros_a, self.sip_zeros_b, self.w.wcs.crpix)

    def wcs2x0(self, wcs):
        """
        decompose a wcs object back into a single vector
        """
        if wcs.sip is None:
            max_size = 10
        else:
            max_size = self.b_ind.max()+1

        x0 = np.zeros(max_size)
        x0[0] = wcs.wcs.crpix[0]
        x0[1] = wcs.wcs.crpix[1]
        x0[2] = wcs.wcs.cdelt[0]
        x0[3] = wcs.wcs.cdelt[1]
        x0[4:8] = wcs.wcs.pc.reshape(4)
        pv = wcs.wcs.get_pv()
        x0[8] = pv[0][2]
        x0[9] = pv[1][2]
        if wcs.sip is not None:
            x0[self.a_ind] = wcs.sip.a.reshape((self.a_order+1.)**2)
            x0[self.b_ind] = wcs.sip.b.reshape((self.b_order+1.)**2)
        return x0

    def return_wcs(self, x0):
        """
        """
        self.set_wcs(x0)
        return self.w

    def __call__(self, x0):
        """
        Parameters
        ----------
        x0 : numpy array
            WCS parameters unrolled into a vector

        Returns
        -------
        The squared-sum distances between the observed and expected positions.
        """
        self.set_wcs(x0)
        # XXX, az alt, or alt az?
        try:
            pix_x, pix_y = self.w.all_world2pix(self.az, self.alt, 0)
        except:
            # if the SIP can't be inverted.
            return np.inf
        # Let's try changing this to a median to help if stars are mis-matched
        resid_sq = np.sum((self.x - pix_x)**2 + (self.y - pix_y)**2)
        return resid_sq


class wcs_zea(wcs_azp):
    def __init__(self, x, y, alt, az, a_order=2, b_order=2, crpix1=0, crpix2=0):
        super(wcs_zea, self).__init__(x, y, alt, az, a_order=a_order,
                                      b_order=b_order, crpix1=crpix1, crpix2=crpix2)
        self.w.wcs.ctype = ["RA---ZEA-SIP", "DEC--ZEA-SIP"]

    def set_wcs(self, x0):
        """
        x0 = [0:crpix1, 1:crpix2, 2:cdelt1, 3:cdelt2, 4:pc, 5:pc, 6:pc, 7:pc, sip... ]
        """
        # Referece Pixel
        self.w.wcs.crpix = [x0[0], x0[1]]

        # Set the cdelt values
        self.w.wcs.cdelt = [x0[2], x0[3]]

        # Set the pc matrix
        self.w.wcs.pc = x0[4:8].reshape((2, 2))

        # Make a new SIP
        if np.size(x0) > 8:
            a = x0[self.a_ind].reshape((self.a_order + 1, self.a_order + 1))
            b = x0[self.b_ind].reshape((self.b_order + 1, self.b_order + 1))
            self.w.sip = Sip(a, b, self.sip_zeros_a, self.sip_zeros_b, self.w.wcs.crpix)

    def wcs2x0(self, wcs):
        """
        decompose a wcs object back into a single vector
        """
        if wcs.sip is None:
            max_size = 10
        else:
            max_size = self.b_ind.max()+1

        x0 = np.zeros(max_size)
        x0[0] = wcs.wcs.crpix[0]
        x0[1] = wcs.wcs.crpix[1]
        x0[2] = wcs.wcs.cdelt[0]
        x0[3] = wcs.wcs.cdelt[1]
        x0[4:8] = wcs.wcs.pc.reshape(4)
        if wcs.sip is not None:
            x0[self.a_ind] = wcs.sip.a.reshape((self.a_order+1.)**2)
            x0[self.b_ind] = wcs.sip.b.reshape((self.b_order+1.)**2)
        return x0


def mag2quasi_dist(mag):
    """Assume all stars have absolute mag of 10.

    Parameters
    ----------
    mag : float or np.array

    Returns
    -------
    dist : float
        Distance to star in parsecs
    """
    # Distance modulus
    mu = mag - 10.
    dist = 10.**(mu/5.+1)
    return dist


class wcs_refine(wcs_azp):
    """Take a catalog of alt,az positions of known stars and minimize the d2 or d3 distance
    """
    def __init__(self, x, y, xy_mag, xy_mjd, ra, dec, rd_mag, location=None,
                 a_order=0, b_order=0, crpix1=0, crpix2=0, alt_limit=15.,
                 what_min='d2'):
        """
        Parameters
        ----------
        x : array
            x-positions of detected objects on the chip
        y : array
            y-positions of detected objects
        xy_mag : array
            magnitude of the detected objects
        xy_mjd : float
            The MJD at the time of the observations
        ra : array
            RA values from a reference catalog
        dec : array
            Dec values from reference catalog
        rd_mag : array
            Magnitude values for reference catalog stars
        what_min : 'd2' or 'd3'
            Should it return the d2 (angular) distances or the d3 (spatial) distances
            (assuming all stars have absolute magnitude of 10)
        """

        self.alt_limit = alt_limit
        self.what_min = what_min
        # Assume all stars are V=10 for distance purposes
        if location is None:
            self.location = EarthLocation(lat=-30.2444*u.degree,
                                     lon=-70.7494*u.degree,
                                     height=2650.0*u.meter)
        else:
            self.location = location

        self.x = x
        self.y = y
        self.dist = mag2quasi_dist(xy_mag)*u.pc

        aa_dist = mag2quasi_dist(rd_mag)
        ref_catalog = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, distance=aa_dist*u.pc)
        self.time = Time(xy_mjd, format='mjd')
        self.ref_catalog = ref_catalog.transform_to(AltAz(obstime=self.time,
                                                          location=self.location))
        self.ref_catalog = self.ref_catalog[np.where(self.ref_catalog.alt.value > alt_limit)]

        # The wcs object we'll be using
        self.w = wcs.WCS(naxis=2)
        self.w.wcs.crpix = [crpix1, crpix2]
        # Fix the reference pixel to zenith
        self.w.wcs.crval = [0, 90]
        self.w.wcs.ctype = ["RA---AZP-SIP", "DEC--AZP-SIP"]

        # Make a sip object with all zero values
        self.a_order = a_order
        self.b_order = b_order
        n_a = int((a_order + 1.)**2)
        n_b = int((b_order + 1)**2)

        self.a_ind = np.arange(n_a) + 10
        self.b_ind = np.arange(n_b) + self.a_ind.max() + 1

        self.sip_zeros_a = np.zeros((a_order+1, a_order + 1))
        self.sip_zeros_b = np.zeros((b_order+1, b_order + 1))
        self.w.sip = None

    def find_distances(self, x0):
        self.set_wcs(x0)
        detected_azs, detected_alts = self.w.all_pix2world(self.x, self.y, 0)
        observed_cat = SkyCoord(detected_azs*u.degree, detected_alts*u.degree,
                                distance=self.dist, frame=AltAz(obstime=self.time, location=self.location))
        try:
            indx, d2, d3 = self.ref_catalog.match_to_catalog_3d(observed_cat)
            result = (indx, d2.value, d3.value)
        except:
            result = ([], np.inf, np.inf)
        return result

    def __call__(self, x0):
        indx, d2, d3 = self.find_distances(x0)
        if self.what_min == 'd2':
            result = np.sum(d2)
        elif self.what_min == 'd3':
            result = np.sum(d3)
        return result


class wcs_refine_zea(wcs_zea):
    """Take a catalog of alt,az positions of known stars and minimize the d3 distance
    """
    def __init__(self, x, y, xy_mag, xy_mjd, ra, dec, rd_mag, location=None,
                 a_order=0, b_order=0, crpix1=0, crpix2=0, alt_limit=15.,
                 what_min='d2', min_func=np.sum):
        """
        Parameters
        ----------
        x : array
            x-positions of detected objects on the chip
        y : array
            y-positions of detected objects
        xy_mag : array
            magnitude of the detected objects
        xy_mjd : float
            The MJD at the time of the observations
        ra : array
            RA values from a reference catalog
        dec : array
            Dec values from reference catalog
        rd_mag : array
            Magnitude values for reference catalog stars
        what_min : 'd2' or 'd3'
            Should it return the d2 (angular) distances or the d3 (spatial) distances
            (assuming all stars have absolute magnitude of 10)
        min_func : function to minimize
            Good options would be np.sum, np.median, np.mean.
        """

        self.alt_limit = alt_limit
        self.what_min = what_min
        # Assume all stars are V=10 for distance purposes
        if location is None:
            self.location = EarthLocation(lat=-30.2444*u.degree,
                                     lon=-70.7494*u.degree,
                                     height=2650.0*u.meter)
        else:
            self.location = location

        self.min_func = min_func
        self.x = x
        self.y = y
        self.dist = mag2quasi_dist(xy_mag)*u.pc

        aa_dist = mag2quasi_dist(rd_mag)
        ref_catalog = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, distance=aa_dist*u.pc)
        self.time = Time(xy_mjd, format='mjd')
        self.ref_catalog = ref_catalog.transform_to(AltAz(obstime=self.time,
                                                          location=self.location))
        self.ref_catalog = self.ref_catalog[np.where(self.ref_catalog.alt.value > alt_limit)]

        # The wcs object we'll be using
        self.w = wcs.WCS(naxis=2)
        self.w.wcs.crpix = [crpix1, crpix2]
        # Fix the reference pixel to zenith
        self.w.wcs.crval = [0, 90]
        self.w.wcs.ctype = ["RA---ZEA-SIP", "DEC--ZEA-SIP"]

        # Make a sip object with all zero values
        self.a_order = a_order
        self.b_order = b_order
        n_a = int((a_order + 1.)**2)
        n_b = int((b_order + 1)**2)

        self.a_ind = np.arange(n_a) + 10
        self.b_ind = np.arange(n_b) + self.a_ind.max() + 1

        self.sip_zeros_a = np.zeros((a_order+1, a_order + 1))
        self.sip_zeros_b = np.zeros((b_order+1, b_order + 1))
        self.w.sip = None

    def find_distances(self, x0):
        self.set_wcs(x0)
        detected_azs, detected_alts = self.w.all_pix2world(self.x, self.y, 0)
        observed_cat = SkyCoord(detected_azs*u.degree, detected_alts*u.degree,
                                distance=self.dist, frame=AltAz(obstime=self.time, location=self.location))
        try:
            indx, d2, d3 = self.ref_catalog.match_to_catalog_3d(observed_cat)
            result = (indx, d2.value, d3.value)
        except:
            result = ([], np.inf, np.inf)
        return result

    def __call__(self, x0):
        indx, d2, d3 = self.find_distances(x0)
        if self.what_min == 'd2':
            result = self.min_func(d2)
        elif self.what_min == 'd3':
            result = self.min_func(d3)
        return result
