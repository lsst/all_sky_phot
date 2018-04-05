import numpy as np
import os
from astropy import units as u
from astropy.coordinates import SkyCoord
from numpy.lib.recfunctions import append_fields


def read_simbad():
    """Read stellar catalog from Simbad
    """

    import lsst.all_sky_phot as temp
    path = os.path.dirname(temp.__file__)
    filenames = ['star_catalog/star_cat_north.dat', 'star_catalog/star_cat_south.dat']
    results = []

    names = ['id', 'ident', 'type', 'coord_string', 'Umag', 'Bmag', 'Vmag',
             'Rmag', 'uMag', 'gMag', 'rMag', 'iMag', 'zMag', 'spect_type', 'note']
    types = [int, '|U50', '|U5', '|U30', float, float, float, float, float,
             float, float, float, float, float, '|U30', int]
    for filename in filenames:
        file = os.path.join(path, filename)
        result = np.genfromtxt(file, dtype=list(zip(names, types)), skip_header=9, delimiter='|',
                               missing_values='~', comments='=')
        # Use astropy to convert the coordinate string to numbers
        coords = SkyCoord(result['coord_string'], unit=(u.hourangle, u.deg))
        result = append_fields(result, ['RA', 'dec'], [coords.ra.value, coords.dec.value])

        results.append(result)

    result = np.concatenate(results)
    return result
