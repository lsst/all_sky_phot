import numpy as np
from astroquery.simbad import Simbad
import matplotlib as plt


# Let's take a few stars from a few frames and try to fit a WCS projection. Then we can try to transform more.

names = ['star_name', 'x', 'y', 'mjd']
types = ['|S10', float, float, float]

obs_stars = np.loadtxt('starcoords.dat', dtype=list(zip(names, types)), skiprows=1, delimiter=',')

ustars = np.unique(obs_stars['star_name'])
names = ['star_name', 'ra', 'dec']
types = ['|S20']*3
star_coords = np.zeros(ustars.size, dtype=list(zip(names, types)))

for i, star_name in enumerate(ustars):
    result = Simbad.query_object(star_name)
    star_coords[i]['star_name'] = star_name
    star_coords[i]['ra'] = result['RA'][0]
    star_coords[i]['dec'] = result['DEC'][0]

# Now to make an array that has x,y,alt,az.


# Fit the projection!
