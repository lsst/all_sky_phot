from __future__ import print_function
import numpy as np
from astroquery.simbad import Simbad
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, ICRS, Longitude, Latitude


def read_manual_stars(filename):
    """
    Read in the stars manually measured from the chip and compute their expected alt and az coordinates.
    Return a numpy array with all the info.
    """

    lsst_location = EarthLocation(lat=-30.2444*u.degree, lon=-70.7494*u.degree, height=2650.0*u.meter)

    names = ['star_name', 'x', 'y', 'mjd']
    types = ['|U10', float, float, float]

    obs_stars = np.loadtxt('starcoords.dat', dtype=list(zip(names, types)), skiprows=1, delimiter=',')

    names.extend(['alt', 'az'])
    types.extend([float, float])

    new_cols = np.zeros(obs_stars.size, dtype=list(zip(names, types)))
    for name in obs_stars.dtype.names:
        new_cols[name] = obs_stars[name]

    obs_stars = new_cols

    ustars = np.unique(obs_stars['star_name'])
    names = ['star_name', 'ra', 'dec']
    types = ['|U20']*3
    star_coords = np.zeros(ustars.size, dtype=list(zip(names, types)))

    # Get the RA,Dec values for each star from Simbad
    for i, star_name in enumerate(ustars):
        result = Simbad.query_object(star_name)
        star_coords[i]['star_name'] = star_name
        ra = result['RA'][0].split(' ')
        ra = ra[0]+'h'+ra[1]+'m'+ra[2]
        star_coords[i]['ra'] = ra
        dec = result['DEC'][0].split(' ')
        dec = dec[0]+'d'+dec[1]+'m'+dec[2]
        star_coords[i]['dec'] = dec

    # Predict the alt, az values for each star at each MJD
    ra = Longitude(star_coords['ra'], unit=u.hourangle)
    dec = Latitude(star_coords['dec'], unit=u.deg)
    star_coords = SkyCoord(ra=ra, dec=dec, frame=ICRS)

    utimes = np.unique(obs_stars['mjd'])
    star_names = []
    mjds = []
    alts = []
    azs = []
    for time in utimes:
        time_mjd = Time(time, format='mjd')
        trans = star_coords.transform_to(AltAz(obstime=time_mjd, location=lsst_location))
        star_names.extend(ustars.tolist())
        mjds.extend([time]*ustars.size)
        alts.extend(trans.alt.value.tolist())
        azs.extend(trans.az.value.tolist())

    names = ['star_name', 'alt', 'az', 'mjd']
    types = ['|U10', float, float, float]

    predicted_array = np.zeros(len(azs), dtype=list(zip(names, types)))
    predicted_array['star_name'] = np.array(star_names)
    predicted_array['alt'] = np.array(alts)
    predicted_array['az'] = np.array(azs)
    predicted_array['mjd'] = np.array(mjds)

    predicted_array.sort(order=['star_name', 'mjd'])
    obs_stars.sort(order=['star_name', 'mjd'])
    hash1 = np.core.defchararray.add(obs_stars['star_name'], obs_stars['mjd'].astype('|U20'))
    hash2 = np.core.defchararray.add(predicted_array['star_name'], predicted_array['mjd'].astype('|U20'))

    good = np.in1d(hash2, hash1)
    predicted_array = predicted_array[good]
    obs_stars['alt'] = predicted_array['alt']
    obs_stars['az'] = predicted_array['az']

    return obs_stars




