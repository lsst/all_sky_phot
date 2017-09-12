import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.time import Time


__all__ = ['lsst_earth_location', 'robustRMS', 'radec2altaz']


def lsst_earth_location():
    """Return the location of LSST as an astropy EarthLocation object.

    XXX--note, should probably eventually be updated to pull this from a database.
    """
    return EarthLocation(lat=-30.2444*u.degree, lon=-70.7494*u.degree, height=2650.0*u.meter)


def robustRMS(x):
    """RMS based on the inter-quartile range.
    """
    iqr = np.percentile(x, 75)-np.percentile(x, 25)
    rms = iqr/1.349  # approximation
    return rms


def radec2altaz(ra, dec, mjd, location=None):
    """I need a stupid converter
    """
    if location is None:
        location = lsst_earth_location()
    time = Time(mjd, format='mjd')
    coords = SkyCoord(ra=ra*u.degree, dec=dec*u.degree)
    ack = coords.transform_to(AltAz(obstime=time, location=location))
    return ack.alt.value, ack.az.value
