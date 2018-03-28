import numpy as np
import matplotlib.pylab as plt
from scipy.optimize import minimize
import glob
import os
import sys
import time
from datetime import datetime
from lsst.all_sky_phot.wcs import wcs_zea, wcs_refine_zea, Fisheye, load_fisheye, distortion_mapper, distortion_mapper_looper
from lsst.all_sky_phot import phot_night, readcr2, readYBC, radec2altaz, star2altaz, phot_image, default_phot_params
from astropy.io import fits
from astropy import wcs
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, ICRS, Longitude, Latitude
import astropy.units as u
from astropy.time import Time, TimezoneInfo
from astropy.table import Table, vstack
from astropy import units as u

from photutils import CircularAperture

from scipy.spatial import KDTree
import healpy as hp
from lsst.sims.utils import healbin


def check_mag_depth(filename='2018-02-05/2018_02_05__20_10_26.fits', outname='result', match_limit=2.5,
                    dao_thresh=6.):
    rough_wcs = load_fisheye('initial_wcs.npz')
    initial_wcs = rough_wcs.wcs

    hdul = fits.open(filename)
    image = hdul[0].data.copy()
    header = hdul[0].header.copy()
    hdul.close()

    utc_offset = 7/24.  # in days.
    lat = Latitude(header['SITELAT'][:-3], unit=u.deg)
    lon = Longitude(header['SITELONG'][:-3], unit=u.deg)
    elevation = 0.728   # km
    PI_backyard = EarthLocation(lat=lat, lon=lon, height=elevation*u.km)

    ybc = readYBC()

    # Set photometry paramters
    phot_params = default_phot_params()
    phot_params['dao_fwhm'] = 2.0
    phot_params['dao_thresh'] = dao_thresh
    phot_table = phot_image(image, phot_params=phot_params)

    phot_appertures = CircularAperture((phot_table['xcenter'], phot_table['ycenter']), r=5.)
    # guess the camera zeropoint
    zp = -18.
    measured_mags = -2.5*np.log10(phot_table['residual_aperture_sum'].data) - zp

    # Calc where we expect stars
    date_string = header['DATE-OBS']
    time_obj = Time(date_string, scale='utc')
    mjd = time_obj.mjd+utc_offset
    alt_cat, az_cat = radec2altaz(ybc['RA'], ybc['Dec'], mjd, location=PI_backyard)
    above = np.where(alt_cat > 5.)
    x_expected, y_expected = initial_wcs.all_world2pix(az_cat[above], alt_cat[above], 0.)
    mag_expected = ybc['Vmag'].values[above]
    apertures = CircularAperture((x_expected, y_expected), r=5.)

    plt.figure(figsize=[20, 20])
    plt.imshow(np.log10(image), cmap='Greys', origin='lower', vmin=2.8, vmax=3.5)
    plt.colorbar()
    plt.xlim([900, 1500])
    plt.ylim([600, 1200])
    # Detected objects in blue
    phot_appertures.plot(color='blue', lw=3, alpha=0.5)
    # Predicted locations in green
    apertures.plot(color='green', lw=3, alpha=0.75)
    plt.title(filename + ', '+header['filters'])
    plt.savefig('Plots/'+outname+'_image.png')
    plt.close()

    bright_limit = 4.
    phot_inbox = np.where((phot_table['xcenter'].value > 900) & (phot_table['xcenter'].value < 1500) &
                          (phot_table['ycenter'].value > 600) & (phot_table['ycenter'].value < 1200))
    predicted_inbox = np.where((x_expected > 900) & (x_expected < 1500) &
                               (y_expected > 600) & (y_expected < 1200) &
                               (mag_expected > bright_limit))

    phot_tree = KDTree(list(zip(x_expected[predicted_inbox], y_expected[predicted_inbox])))
    distances, indices = phot_tree.query(np.array((phot_table['xcenter'].value[phot_inbox],
                                                   phot_table['ycenter'].value[phot_inbox])).T)

    matched = indices[np.where(distances < match_limit)]

    cumulative = np.arange(x_expected[predicted_inbox].size)+1
    detected = np.zeros(x_expected[predicted_inbox].size)
    detected[matched] += 1

    order = np.argsort(mag_expected[predicted_inbox])

    detected = np.cumsum(detected[order])

    dash_limit = 0.75
    try:
        mag_limit = mag_expected[predicted_inbox][order][np.max(np.where(detected/cumulative > dash_limit))]
    except:
        mag_limit = -1

    check_mag = 6.
    cml = np.max(np.where(mag_expected[predicted_inbox][order] < check_mag))

    plt.figure()
    plt.plot(mag_expected[predicted_inbox][order], cumulative)
    plt.plot(mag_expected[predicted_inbox][order], detected)
    plt.axvline(mag_limit, linestyle='--')
    plt.xlabel('V mag')
    plt.ylabel('N')
    plt.title('detecting %.0f%% of stars at V mag=%.1f' % (dash_limit*100, mag_limit))
    frac = detected/cumulative
    plt.axvline(mag_expected[predicted_inbox][order][cml], color='r',
                linestyle='--', label='%.0f%% at %.1f' % (frac[cml]*100, check_mag))
    plt.legend()
    plt.savefig('Plots/'+outname+'_detection.png')
    plt.close()

if __name__ == '__main__':

    Lfiles = ['2018_02_05__21_23_56.fits', '2018_02_05__21_25_15.fits',
              '2018_02_05__21_26_37.fits', '2018_02_05__21_27_55.fits', '2018_02_05__21_22_37.fits']

    for i, filename in enumerate(Lfiles):
        check_mag_depth('2018-02-05/'+ filename, outname='L_%i' % i, match_limit=8.5, dao_thresh=7.)


    gfiles = ['2018_02_05__20_10_26.fits', '2018_02_05__20_11_45.fits', '2018_02_05__20_13_05.fits',
              '2018_02_05__20_14_25.fits', '2018_02_05__20_15_44.fits', '2018_02_05__20_17_03.fits']
    for i, filename in enumerate(gfiles):
        check_mag_depth('2018-02-05/'+ filename, outname='g_%i' % i)

    ifiles = ['2018_02_05__20_52_58.fits', '2018_02_05__20_54_17.fits', '2018_02_05__20_55_36.fits',
              '2018_02_05__20_56_57.fits', '2018_02_05__20_58_17.fits']

    for i, filename in enumerate(ifiles):
        check_mag_depth('2018-02-05/'+ filename, outname='i_%i' % i, dao_thresh=4.)

    
