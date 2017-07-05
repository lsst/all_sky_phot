import numpy as np
import healpy as hp


class small_shift_find(object):
    """See if there is a small shift that needs to be applied on top of the wcs.
    """
    def __init__(self, observed_x, observed_y, observed_mag, maxshift=5):

        self.maxshift = maxshift
        # Make a kdtree for the observed points
        pass

    def distance_median(self, x0):



    def __call__(self, catalog_x, catalog_y, catalog_mag):


        return shift_x, shift_y




def calc_extinction(image, wcs, zp, catalog_alt, catalog_az, catalog_mag, nside=32):
    """
    Generate a map of the extinction on the sky

    Parameters
    ----------
    image : array
        The image to find the exticntion map from
    wcs : wcs-like object
        WCS describing the image
    zp : float
        The zeropoint of the image
    catalog_alt : array
        Altitude of stars expected to be in the image (degrees)
    catalog_az : array
        Azimuth of stars expected in the image (degrees)
    catalog_mag : array
        Magnitudes of stars expected in the image
    nside : int
        Healpixel nside to set resoltion of output map

    Output
    ------
    extinction : array
        A healpixel array (where latitude and longitude are altitude and azimuth). Pixels
        with no stars are filled with the healpixel mask value. Non-masked pixesl have the
        measured extinction in magnitudes.
    """

    
    # Find the healpixel for each catalog star
    lat = np.radians(90.-catalog_alt)
    catalog_hp = hp.ang2pix(nside, lat, np.radians(catalog_az))

    order = np.argsort(catalog_hp)
    catalog_alt = catalog_alt[order]
    catalog_az = catalog_az[order]
    catalog_mag = catalog_mag[order]
    catalog_hp = catalog_hp[order]
    catalog_x, catalog_y = wcs.all_world2pix(catalog_az, catalog_alt, 0)


    # Run detection on the image
    
    


    result = np.empty(hp.nside2npix(nside), dtype=float)
    result.fill(hp.UNSEEN)
    uhp = np.unique(catalog_hp)
    # Loop over each healpixel, compute the extinction
    for healpix_id in uhp:
        




