import numpy as np
from astropy.io import fits
from astropy import wcs
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, ICRS, Longitude, Latitude
import astropy.units as u
from astropy.time import Time
from astropy.table import Table, vstack

# restore list of photometry tables, one per image
temp = np.load('full_night.npz')
phot_tables = temp['phot_tables'][()]
temp.close()

hdulist = fits.open('initial_wcs.fits')
w = wcs.WCS(hdulist[0].header)
hdulist.close()

lsst_location = EarthLocation(lat=-30.2444*u.degree, lon=-70.7494*u.degree, height=2650.0*u.meter)


def coord_trans(ptable, w):
    """
    Take a photometry table and add ra and dec cols
    """
    time = Time(ptable['mjd'].max(), format='mjd')
    az, alt = w.all_pix2world(ptable['xcenter'], ptable['ycenter'], 0)
    coords = AltAz(az=az*u.degree, alt=alt*u.degree, location=lsst_location, obstime=time)
    sky_coords = coords.transform_to(ICRS)
    ptable['alt_rough'] = coords.alt
    ptable['az_rough'] = coords.az
    ptable['ra_rough'] = sky_coords.ra
    ptable['dec_rough'] = sky_coords.dec


for ptable in phot_tables[1:]:
    ptable = coord_trans(ptable, w)

phot_table = vstack(phot_tables)


# np.savez('phot_tables_rough_wcs.npz', phot_tables=phot_tables)

