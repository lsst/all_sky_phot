import numpy as np
from astropy.wcs import Sip
from astropy import wcs

# Let's make a new attempt to fit things, this time using the WCS object.
__all__ = ['wcs_azp']


class wcs_azp(object):

    def __init__(self, x, y, alt, az, a_order=2, b_order=2, crpix1=0, crpix2=0):
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

        self.az = az
        self.alt = alt
        self.x = x
        self.y = y

        # The wcs object we'll be using
        self.w = wcs.WCS(naxis=2)
        self.w.wcs.crpix = [crpix1, crpix2]
        # Fix the reference pixel to zenith
        self.w.wcs.crval = [0, 90]
        self.w.wcs.ctype = ["RA---ZEA", "DEC--ZEA"]
        #self.w.wcs.ctype = ["RA---AZP-SIP", "DEC--AZP-SIP"]

        self.world_coords = np.vstack((az, alt))
        self.pix_coords = np.vstack((x, y))
        # Make a sip object with all zero values
        self.a_order = a_order
        self.b_order = b_order
        n_a = int((a_order + 1.)**2)
        n_b = int((b_order + 1)**2)

        self.a_ind = np.arange(n_a) + 10
        self.b_ind = np.arange(n_b) + self.a_ind.max() + 1

        self.sip_zeros_a = np.zeros((a_order+1, a_order + 1))
        self.sip_zeros_b = np.zeros((b_order+1, b_order + 1))
        self.w.sip = Sip(np.zeros((a_order+1, a_order + 1)), np.zeros((b_order+1, b_order + 1)),
                         np.zeros((a_order+1, a_order + 1)), np.zeros((b_order+1, b_order + 1)),
                         [0, 0])

    def set_wcs(self, x0):
        """
        x0 = [cdelt1, cdelt2, ]
        """
        # 
        self.w.wcs.crpix = [x0[0], x0[1]]

        # Set the cdelt values
        self.w.wcs.cdelt = [x0[2], x0[3]]

        # Set the pc matrix
        #self.w.wcs.pc = x0[4:8].reshape((2, 2))

        # Set mu and gamma
        #self.w.wcs.set_pv([(1, 0, x0[8]), (1, 1, x0[9])])

        # Make a new SIP
        #a = x0[self.a_ind].reshape((self.a_order + 1, self.a_order + 1))
        #b = x0[self.b_ind].reshape((self.b_order + 1, self.b_order + 1))
        #self.w.sip = Sip(a, b, self.sip_zeros_a, self.sip_zeros_b, [0, 0])

        # Temp turn off SIP
        self.w.sip = None

    def return_wcs(self, x0):
        self.set_wcs(x0)
        return self.w

    def __call__(self, x0):
        """
        
        """
        self.set_wcs(x0)
        # XXX, az alt, or alt az?
        pix_x, pix_y = self.w.all_world2pix(self.az, self.alt, 0)
        resid_sq = np.sum((self.x - pix_x)**2 + (self.y - pix_y)**2)
        return resid_sq
