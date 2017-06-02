import numpy as np
from astropy.wcs import Sip
from astropy import wcs

# Let's make a new attempt to fit things, this time using the WCS object.
__all__ = ['wcs_azp', 'wcs_zea']





class wcs_azp(object):

    def __init__(self, x, y, alt, az, a_order=2, b_order=2, crpix1=0, crpix2=0, nparams=10):
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
        #self.w.wcs.ctype = ["RA---ZEA", "DEC--ZEA"]
        self.w.wcs.ctype = ["RA---AZP-SIP", "DEC--AZP-SIP"]

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
        self.w.sip = None

    def set_wcs(self, x0):
        """
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

        x0 = np.zeros(self.b_ind.max()+1)

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
        self.set_wcs(x0)
        return self.w

    def __call__(self, x0):
        """
        
        """
        self.set_wcs(x0)
        # XXX, az alt, or alt az?
        try:
            pix_x, pix_y = self.w.all_world2pix(self.az, self.alt, 0)
        except:
            # if the SIP can't be inverted.
            return np.inf
        resid_sq = np.sum((self.x - pix_x)**2 + (self.y - pix_y)**2)
        return resid_sq


class wcs_zea(wcs_azp):
    def __init__(self, x, y, alt, az, a_order=2, b_order=2, crpix1=0, crpix2=0, nparams=8):
        super(wcs_zea, self).__init__(x, y, alt, az, a_order=a_order,
                                      b_order=b_order, crpix1=crpix1, crpix2=crpix2,
                                      nparams=nparams)
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
        if np.size(x0) > 10:
            a = x0[self.a_ind].reshape((self.a_order + 1, self.a_order + 1))
            b = x0[self.b_ind].reshape((self.b_order + 1, self.b_order + 1))
            self.w.sip = Sip(a, b, self.sip_zeros_a, self.sip_zeros_b, self.w.wcs.crpix)
