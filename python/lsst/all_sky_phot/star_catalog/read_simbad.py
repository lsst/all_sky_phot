import numpy as np
import os
from astropy import units as u
from astropy.coordinates import SkyCoord
from numpy.lib.recfunctions import append_fields
from scipy.spatial import cKDTree as kdtree
import sys


# Ugh, copied from sims_featureScheduler. Should put kdtree utils in sims_utils or something
def treexyz(ra, dec):
    """
    Utility to convert RA,dec postions in x,y,z space, useful for constructing KD-trees.

    Parameters
    ----------
    ra : float or array
        RA in radians
    dec : float or array
        Dec in radians

    Returns
    -------
    x,y,z : floats or arrays
        The position of the given points on the unit sphere.
    """
    # Note ra/dec can be arrays.
    x = np.cos(dec) * np.cos(ra)
    y = np.cos(dec) * np.sin(ra)
    z = np.sin(dec)
    return x, y, z


def rad_length(radius=1.75):
    """
    Convert an angular radius into a physical radius for a kdtree search.

    Parameters
    ----------
    radius : float
        Radius in degrees.
    """
    x0, y0, z0 = (1, 0, 0)
    x1, y1, z1 = treexyz(np.radians(radius), 0)
    result = np.sqrt((x1-x0)**2+(y1-y0)**2+(z1-z0)**2)
    return result


def read_simbad(isolate_catalog=False, isolate_radius=20., isolate_mag=3., leafsize=100):
    """Read stellar catalog from Simbad

    Parameters
    ----------
    isolate_catalog : bool (False)
        Throw out stars that have neighbors that are too bright
    isolate_radius : float (20.)
        How isolated to demand a star to be in arcmin.
    isolate_mag : flat (3.)

    Returns
    -------
    numpy array of stellar properties. RA and dec in degrees

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

    if isolate_catalog:
        # list to store if we pass isolation conditions
        is_isolated = []
        # Convert ra, dec to x,y,z
        x, y, z = treexyz(np.radians(result['RA']), np.radians(result['dec']))

        search_radius = rad_length(radius=isolate_radius/60.)
        maxI = float(x.size)
        tree = kdtree(list(zip(x, y, z)), leafsize=leafsize, balanced_tree=False, compact_nodes=False)
        for i, xx in enumerate(x):
            neighbors_indices = tree.query_ball_point((x[i], y[i], z[i]), search_radius)
            to_bright = np.where(result['Vmag'][neighbors_indices]-isolate_mag < result['Vmag'][i])[0]
            # Should match itself, so if more than 1

            if np.size(to_bright) > 1:
                is_isolated.append(False)
            else:
                is_isolated.append(True)
            #progress = i/maxI
            #text = "\rprogress = %.1f%%"%progress*100
            #sys.stdout.write(text)
            #sys.stdout.flush()
        result = result[is_isolated]

    return result
