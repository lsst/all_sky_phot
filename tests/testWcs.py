import numpy as np
import unittest
from lsst.all_sky_phot.wcs import Fisheye
import lsst.utils.tests
from astropy import wcs


class TestFeatures(unittest.TestCase):

    def testWCS(self):
        # Make sure a reasonable WCS round-trips
        dummyWCS = wcs.WCS(naxis=2)
        dummyWCS.wcs.ctype = ["RA---ZEA-SIP", "DEC--ZEA-SIP"]
        x0 = np.array([2.87356521e+03, 1.98559533e+03, 1., 1., .036,
                      0.0027, 0.00295, -0.0359])
        dummyWCS.wcs.crpix[0] = x0[0]
        dummyWCS.wcs.crpix[1] = x0[1]
        dummyWCS.wcs.cdelt[0] = x0[2]
        dummyWCS.wcs.cdelt[1] = x0[3]
        dummyWCS.wcs.pc = x0[4:8].reshape((2, 2))

        x, y = np.meshgrid(np.arange(1000., 2000, 100), np.arange(1000., 2000, 100))
        az, alt = dummyWCS.all_pix2world(x, y, 0)

        # just as a first step, make sure astropy round trips
        xp, yp = dummyWCS.all_world2pix(az, alt, 0)

        np.testing.assert_array_almost_equal(x, xp)
        np.testing.assert_array_almost_equal(y, yp)

        # Make a simple distorion map
        fish_x, fish_y = np.meshgrid(np.arange(500., 2500, 100), np.arange(500., 2500, 100))
        dist_x = 0.001*fish_x + 2 - 0.001*fish_y
        dist_y = 0.001*fish_x - 2. + 0.001*fish_y

        # Create a fisheye WCS object
        fish_wcs = Fisheye(dummyWCS, fish_x.flatten(), fish_y.flatten(),
                           dist_x.flatten(), dist_y.flatten())

        # Round trip from xy, to alt, az and back
        alt, az = fish_wcs.all_pix2world(x, y, 0.)
        xp, yp = fish_wcs.all_world2pix(az, alt, 0.)
        np.testing.assert_array_almost_equal(x, xp)
        np.testing.assert_array_almost_equal(y, yp)






class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
