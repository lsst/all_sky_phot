#!/Users/yoachim/lsst/DarwinX86/miniconda2/3.19.0.lsst4/bin/python

import argparse


import numpy as np
import matplotlib.pylab as plt
from scipy.optimize import minimize
import glob
import os
import sys

from read_ybc import readYBC
from read_stars import read_manual_stars
from wcs_utils import apply_wcs_to_photometry, radec2altaz

from lsst.all_sky_phot.wcs import wcs_zea, wcs_refine_zea, fit_xyshifts, Fisheye, distortion_mapper
from lsst.all_sky_phot import phot_night, readcr2

from astropy import wcs
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, ICRS, Longitude, Latitude
import astropy.units as u
from astropy.time import Time
from astropy.table import Table, vstack
from astropy import units as u

from photutils import CircularAperture


def run_map(minindx, maxindx, filename='out.npz'):
    # Now to run photometry on a full image and refine the WCS solution
    # Load the Yale bright star catalog
    from astropy.io import fits
    ybc = readYBC()


    # Load up the photometry tables from the 1st night
    temp = np.load('012716_night_phot.npz')
    phot_tables = temp['phot_tables'][()]
    temp.close()

    # read in the rough WCS
    # the simple wcs based on just identified stars
    hdulist = fits.open('wcs_refined.fits')
    wcs_refined = wcs.WCS(hdulist[0].header)
    hdulist.close()

    # Generate cataog, alt, az, mjd.
    alts = []
    azs = []
    mjds = []
    observed_x = []
    observed_y = []
    observed_mjd = []
    lsst_location = EarthLocation(lat=-30.2444*u.degree, lon=-70.7494*u.degree, height=2650.0*u.meter)
    for phot_table in phot_tables[minindx:maxindx]:
        mjd = phot_table['mjd'][0]
        alt, az = radec2altaz(ybc['RA'], ybc['Dec'], mjd, location=lsst_location)
        good = np.where(alt > 0.)
        alts.append(alt[good])
        azs.append(az[good])
        mjds.append(az[good]*0+mjd)
        
        observed_x.append(phot_table['xcenter'].value)
        observed_y.append(phot_table['ycenter'].value)
        observed_mjd.append(phot_table['mjd'].data)

    alts = np.concatenate(alts)
    azs = np.concatenate(azs)
    mjds = np.concatenate(mjds)

    observed_x = np.concatenate(observed_x)
    observed_y = np.concatenate(observed_y)
    observed_mjd = np.concatenate(observed_mjd)

    # (3870, 5796)
    xgrid = np.linspace(0,5796, 35)
    ygrid = np.linspace(0,3870, 25)
    xgrid, ygrid = np.meshgrid(xgrid, ygrid)

    xp = []
    yp = []
    fits = []
    i=0
    imax = float(xgrid.size*ygrid.size)
    for u_center in xgrid.ravel():
        for v_center in ygrid.ravel():
            xp.append(u_center)
            yp.append(v_center)
            fit_result = distortion_mapper(observed_x, observed_y, observed_mjd, alts, azs,
                                           mjds, wcs_refined,window=100, u_center=u_center, v_center=v_center)
            fits.append(fit_result)
            i += 1
            progress = i/imax*100
            text = "\rprogress = %.2f%%" % progress
            sys.stdout.write(text)
            sys.stdout.flush()
    np.savez(filename, xp=xp, yp=yp, fits=fits)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--minindx", type=int, default=0, help="")
    parser.add_argument("--maxindx", type=int, default=50, help="")

    args = parser.parse_args()

    run_map(args.minindx, args.maxindx, filename='fit_result_%i_%i' % (args.minindx, args.maxindx))



