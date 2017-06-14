import numpy as np
from astropy.io import fits
from astropy import wcs
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, ICRS, Longitude, Latitude
import astropy.units as u
from astropy.time import Time
from astropy.table import Table, vstack
import matplotlib.pylab as plt
from read_ybc import readYBC
import lsst.all_sky_phot.wcs as asp
from scipy.optimize import minimize

# Some handy utilities for bootstrapping WCS solutions


lsst_location = EarthLocation(lat=-30.2444*u.degree,
                              lon=-70.7494*u.degree,
                              height=2650.0*u.meter)


def radec2altaz(ra, dec, mjd, location=lsst_location):
    """I need a stupid converter
    """
    time = Time(mjd, format='mjd')
    coords = SkyCoord(ra=ra*u.degree, dec=dec*u.degree)
    ack = coords.transform_to(AltAz(obstime=time, location=location))
    return ack.alt.value, ack.az.value


def apply_wcs_to_photometry(ptable, w, location=lsst_location):
    """
    Take a photometry table and add ra and dec cols

    Parameters
    ----------
    ptable : astropy table
        Needs columns xcenter, ycenter, and mjd.
        Assumes all the mjd are the same
    wcs : wcs object
        the World Coordinate System object
    location : astropy EarthLocation object

    Returns
    -------
    Photometry table with columns added for the alt, az, ra, dec
    """
    time = Time(ptable['mjd'].max(), format='mjd')
    az, alt = w.all_pix2world(ptable['xcenter'], ptable['ycenter'], 0)
    coords = AltAz(az=az*u.degree, alt=alt*u.degree, location=location, obstime=time)
    sky_coords = coords.transform_to(ICRS)
    ptable['alt_wcs'] = coords.alt
    ptable['az_wcs'] = coords.az
    ptable['ra_wcs'] = sky_coords.ra
    ptable['dec_wcs'] = sky_coords.dec
    return ptable


def match_catalog(ptable, catalog, location=lsst_location):
    """Match the photometry to a catalog

    Add matched_alt, az, ra, dec columns. Maybe also matched star ID
    """

    phot_cat = SkyCoord(ra=ptable['ra_wcs'].value*u.degree, dec=ptable['dec_wcs'].value*u.degree)
    idx, d2d, d3d = phot_cat.match_to_catalog_sky(catalog)

    ptable['ra_matched'] = catalog.ra[idx]
    ptable['dec_matched'] = catalog.dec[idx]

    catalog.transform_to(AltAz, obstime=phot_cat['mjd'].max(), location=location)
    ptable['alt_matched'] = catalog.alt[idx]
    ptable['az_matched'] = catalog.az[idx]
    ptable['d2d'] = d2d

    return ptable



