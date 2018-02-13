import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, ICRS, Longitude, Latitude
from astropy.time import Time
from astroquery.simbad import Simbad


__all__ = ['lsst_earth_location', 'robustRMS', 'radec2altaz', 'star2altaz']


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


def star2altaz(names, mjds, location='lsst'):
    """Convert star name and mjd to predicted alt and az
    """

    if location == 'lsst':
        location = EarthLocation(lat=-30.2444*u.degree, lon=-70.7494*u.degree, height=2650.0*u.meter)

    dnames = ['star_name', 'mjd', 'alt', 'az']
    types = ['|U20', float, float, float]
    results = np.zeros(np.size(names), dtype=list(zip(dnames, types)))

    # Use Simbad to look up the position of all the stars
    coord_dict = {}
    u_names = np.unique(names)
    for star_name in u_names:
        simbad_result = Simbad.query_object(star_name)
        ra = simbad_result['RA'][0].split(' ')
        ra = ra[0]+'h'+ra[1]+'m'+ra[2]
        dec = simbad_result['DEC'][0].split(' ')
        dec = dec[0]+'d'+dec[1]+'m'+dec[2]
        coord_dict[star_name] = (ra, dec)

    for i, star_name in enumerate(names):
        ra = Longitude(coord_dict[star_name][0], unit=u.hourangle)
        dec = Latitude(coord_dict[star_name][1], unit=u.deg)
        star_coord = SkyCoord(ra=ra, dec=dec, frame=ICRS)
        time_mjd = Time(mjds[i], format='mjd')
        trans = star_coord.transform_to(AltAz(obstime=time_mjd, location=location))
        results[i]['star_name'] = star_name
        results[i]['mjd'] = mjds[i]
        results[i]['alt'] = trans.alt.value
        results[i]['az'] = trans.az.value

    return results

