import numpy as np
import glob
from lsst.all_sky_phot.wcs import wcs_zea, wcs_refine_zea, Fisheye, distortion_mapper, distortion_mapper_looper, load_fisheye
from lsst.all_sky_phot import phot_night, readcr2, readYBC, radec2altaz, star2altaz, phot_image, default_phot_params
from lsst.all_sky_phot.star_catalog import read_simbad
from astropy.io import fits
from astropy.coordinates import EarthLocation, Longitude, Latitude
import astropy.units as u
from astropy.time import Time
from astropy.stats import sigma_clipped_stats
from photutils import CircularAperture, CircularAnnulus, aperture_photometry

utc_offset = 7/24.  # in days.


def l3_phot(filename, L3_wcs, bsc):
    # read in a file and return the photometry table
    hdul = fits.open(filename)
    image = hdul[0].data.copy()
    header = hdul[0].header.copy()

    # grab the location from the header
    lat = Latitude(hdul[0].header['SITELAT'][:-3], unit=u.deg)
    lon = Longitude(hdul[0].header['SITELONG'][:-3], unit=u.deg)
    elevation = 0.728   # km
    PI_backyard = EarthLocation(lat=lat, lon=lon, height=elevation*u.km)
    hdul.close()

    date_string = header['DATE-OBS']
    time_obj = Time(date_string, scale='utc')
    mjd = time_obj.mjd+utc_offset
    alt_cat, az_cat = radec2altaz(bsc['RA'], bsc['dec'], mjd, location=PI_backyard)

    good_cat = np.where((alt_cat > 18.1) & (bsc['Vmag'] < 7.5) & (bsc['Vmag'] > 1.))[0]
    x_expected, y_expected = L3_wcs.all_world2pix(az_cat[good_cat], alt_cat[good_cat], 0.)
    # clean out any random nan's
    goodpix = np.where((~np.isnan(x_expected)) & (~np.isnan(y_expected)))[0]
    good_cat = good_cat[goodpix]
    x_expected = x_expected[goodpix]
    y_expected = y_expected[goodpix]
    apertures = CircularAperture((x_expected, y_expected), r=5.)
    bkg_aper = CircularAnnulus((x_expected, y_expected), r_in=7., r_out=10.)

    forced_table = aperture_photometry(image, [apertures, bkg_aper])
    # simple sky subtraction
    bkg_mean = forced_table['aperture_sum_1'] / bkg_aper.area()
    bkg_sum = bkg_mean * apertures.area()
    final_sum = forced_table['aperture_sum_0'] - bkg_sum
    forced_table['residual_aperture_sum'] = final_sum

    # Let's get the median sigma-clipped backgrounds
    # https://github.com/astropy/astropy-workshop/blob/master/09-Photutils/photutils_local_backgrounds.ipynb
    bkg_mask = bkg_aper.to_mask(method='center')
    bkg_median = []
    for mask in bkg_mask:
        aper_data = mask.multiply(image)
        aper_data = aper_data[mask.data > 0]

        # perform a sigma-clipped median
        _, median_sigclip, _ = sigma_clipped_stats(aper_data)
        bkg_median.append(median_sigclip)

    bkg_median = np.array(bkg_median)
    # correct for aperture area, subtract the background, and add table columns
    forced_table['annulus_median'] = bkg_median
    forced_table['aperture_bkg2'] = bkg_median * apertures.area()
    forced_table['aperture_sum_bkgsub2'] = forced_table['aperture_sum_0'] - forced_table['aperture_bkg2']
    # Record which star and it's expected alt and az
    forced_table['catalog_indx'] = good_cat
    forced_table['az'] = az_cat[good_cat]
    forced_table['alt'] = alt_cat[good_cat]

    return forced_table

if __name__ == '__main__':
    L3_wcs = load_fisheye('L3_wcs_w_shift.npz')
    bsc = read_simbad(isolate_catalog=True, isolate_radius=20.)
    filelist = glob.glob('2018-02-05/*.fits')
    filelist.extend(glob.glob('2018-01-26/*.fits'))
    filelist = np.array(filelist)

    # Check the filter names
    filters = []
    for filename in filelist:
        hdul = fits.open(filename)
        filters.append(hdul[0].header['FILTERS'])
        hdul.close()
    good = np.where(np.array(filters) == 'Filter_L3')
    filenames = filelist[good[0]]
    phot_tables = []
    for filename in filenames:
        phot_tables.append(l3_phot(filename, L3_wcs, bsc))

    np.savez('L3phot.npz', phot_tables=phot_tables)
