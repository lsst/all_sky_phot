import numpy as np
from astropy.modeling.projections import Sky2Pix_ZEA, AffineTransformation2D, Sky2Pix_AZP
from astropy.wcs import Sip

__all__ = ['AZP_SIP', 'AZP_affine', 'ZEA_affine']


class AZP_SIP(object):
    """
    Given a catalog of stars with measured chip positions x,y and known alt,az values,
    fit the alt,az to x,y solution using the AZP projeciton, SIP distortions, and affine transformation
    """
    def __init__(self, x, y, alt, az, ap_order=2, bp_order=2):
        """
        Parameters
        ----------
        x : array (float)
            x-positions on the chip
        y : array (float)
            y positions on the chip
        alt : array (float)
            Altitudes of the stars (degrees)
        az : array (float)
            Azimuths of the stars (degrees)

        """
        self.projection = Sky2Pix_AZP()
        self.affine = AffineTransformation2D()
        self.sip = Sip
        self.x = x
        self.y = y
        self.alt = alt
        self.az = az
        self.ap_order = ap_order
        self.bp_order = bp_order
        self.ap_size = (self.ap_order+1)**2
        self.bp_size = (self.bp_order+1)**2
        self.ap_ind = np.arange(self.ap_size)
        self.bp_ind = np.arange(self.bp_size) + self.ap_ind.max()+1
        self.affine_ind = np.arange(6) + self.bp_ind.max() + 1
        self.crpix = np.zeros(2)
        self.zs = np.zeros((3, 3))

    def altaz2xy(self, x0):
        """
        """
        ap = x0[self.ap_ind].reshape((self.ap_order + 1, self.ap_order + 1))
        bp = x0[self.bp_ind].reshape((self.bp_order + 1, self.bp_order + 1))
        sip = self.sip(self.zs, self.zs, ap, bp, self.crpix)

        # Project to plane
        self.projection.mu = x0[-2]
        self.projection.gamma = x0[-1]
        newx, newy = self.projection(self.az, self.alt)

        # Apply SIP distorions
        new_coords = sip.foc2pix(np.vstack((newx, newy)).T, 0)
        newx = new_coords[:, 0]
        newy = new_coords[:, 1]

        # Apply affine transformation
        self.affine.translation.value = x0[self.affine_ind[0:2]]
        self.affine.matrix.value = x0[self.affine_ind[2:]]
        newx, newy = self.affine(newx, newy)

        return newx, newy

    def __call__(self, x0):
        """
        Parameters
        ----------
        x0 : array
            x0[0:n] : ap matrix values
            x0[n:m] : bp matric values
            x0[m:m+6] : affine transformation values
            x0[-2] : AZP mu
            x0[-1] : AZP gamma

        Returns
        -------
        residual : float
            The sum squared distances between the projected and actual pixel coordinates
        """
        newx, newy = self.altaz2xy(x0)
        residual = np.sum((newx - self.x)**2 + (newy-self.y)**2)
        return residual


class AZP_affine(object):
    def __init__(self, x, y, alt, az):
        self.projection = Sky2Pix_AZP()
        self.affine = AffineTransformation2D()
        self.x = x
        self.y = y
        self.alt = alt
        self.az = az

    def compute(self, x0):
        self.affine.translation.value = x0[0:2]
        self.affine.matrix.value = x0[2:6]
        self.projection.mu = x0[6]
        self.projection.gamma = x0[7]
        newx, newy = self.projection(self.az, self.alt)
        newx, newy = self.affine(newx, newy)
        return newx, newy

    def __call__(self, x0):
        newx, newy = self.compute(x0)
        residual = np.sum((newx - self.x)**2 + (newy-self.y)**2)
        return residual


class ZEA_affine(object):
    def __init__(self, x, y, alt, az):
        self.projection = Sky2Pix_ZEA()
        self.affine = AffineTransformation2D()
        self.x = x
        self.y = y
        self.projx, self.projy = self.projection(az, alt)

    def compute(self, x0):
        self.affine.translation.value = x0[0:2]
        self.affine.matrix.value = x0[2:]
        newx, newy = self.affine(self.projx, self.projy)
        return newx, newy

    def __call__(self, x0):
        """
        x0 : numpy.array
            x-trans, ytrans, affine[0,0], [0,1], [1,0], [1,1]
        """
        newx, newy = self.compute(x0)

        residual = np.sum((newx - self.x)**2 + (newy-self.y)**2)
        return residual

