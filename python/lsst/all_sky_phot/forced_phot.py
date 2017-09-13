import numpy as np
import healpy as hp
import photutils as phu
from scipy.stats import binned_statistic
from lsst.all_sky_phot.phot_night import default_phot_params
from astropy.stats import sigma_clipped_stats


__all__ = ['forced_phot']


def forced_phot(image, wcs, zp, catalog_alt, catalog_az, catalog_mag, catalog_id,
                nside=8, phot_params=None, do_background=False, return_table=False, mjd=0.):
    """
    Generate a map of the extinction on the sky

    Parameters
    ----------
    image : array
        The image to find the exticntion map from
    wcs : wcs-like object
        WCS describing the image
    zp : float
        The zeropoint of the image
    catalog_alt : array
        Altitude of stars expected to be in the image (degrees)
    catalog_az : array
        Azimuth of stars expected in the image (degrees)
    catalog_mag : array
        Magnitudes of stars expected in the image
    nside : int
        Healpixel nside to set resoltion of output map
    do_background : bool (False)
        Do a 2D background model subtraction. Skipped by default because astropy is really slow.
    return_table : bool (False)
        If True, return the astropy table with the photometry, otherwise, return the extinction map.

    Output
    ------
    extinction : array
        A healpixel array (where latitude and longitude are altitude and azimuth). Pixels
        with no stars are filled with the healpixel mask value. Non-masked pixesl have the
        measured extinction in magnitudes.
    """

    if phot_params is None:
        phot_params = default_phot_params()

    # Find the healpixel for each catalog star
    lat = np.radians(90.-catalog_alt)
    catalog_hp = hp.ang2pix(nside, lat, np.radians(catalog_az))

    order = np.argsort(catalog_hp)
    catalog_alt = catalog_alt[order]
    catalog_az = catalog_az[order]
    catalog_mag = catalog_mag[order]
    catalog_id = catalog_id[order]
    catalog_hp = catalog_hp[order]
    catalog_x, catalog_y = wcs.all_world2pix(catalog_az, catalog_alt, 0)

    good_transform = ~np.isnan(catalog_x)

    # Find the x,y positions of helpix centers
    lat, lon = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))
    hp_alt = np.degrees(np.pi/2. - lat)
    hp_az = np.degrees(lon)
    hp_x, hp_y = wcs.all_world2pix(hp_az, hp_alt, 0)

    # Run the photometry at the expected catalog positions
    if do_background:
        sigma_clip = phu.SigmaClip(sigma=phot_params['bk_clip_sigma'], iters=phot_params['bk_iter'])
        bkg_estimator = phu.MedianBackground()
        bkg = phu.Background2D(image, (phot_params['background_size'], phot_params['background_size']),
                               filter_size=(phot_params['bk_filter_size'], phot_params['bk_filter_size']),
                               sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
        bk_img = image - bkg.background
    else:
        bk_img = image

    positions = list(zip(catalog_x[good_transform], catalog_y[good_transform]))
    apertures = phu.CircularAperture(positions, r=phot_params['apper_r'])
    annulus_apertures = phu.CircularAnnulus(positions, r_in=phot_params['ann_r_in'],
                                            r_out=phot_params['ann_r_out'])

    apers = [apertures, annulus_apertures]
    phot_table = phu.aperture_photometry(bk_img, apers)

    bkg_mean = phot_table['aperture_sum_1'] / annulus_apertures.area()
    bkg_sum = bkg_mean * apertures.area()
    final_sum = phot_table['aperture_sum_0'] - bkg_sum
    phot_table['residual_aperture_sum'] = final_sum
    phot_table['catalog_id'] = catalog_id[good_transform]

    phot_table['residual_aperture_mag'] = -2.5*np.log10(final_sum) + zp
    detected = np.where(final_sum > 0)

    mag_difference = phot_table['residual_aperture_mag'][detected] - catalog_mag[good_transform][detected]
    phot_table['mag_difference'] = np.zeros(phot_table['residual_aperture_mag'].data.size, dtype=float)
    phot_table['mjd'] = np.zeros(phot_table['residual_aperture_mag'].data.size, dtype=float)
    phot_table['mag_difference'][detected] = mag_difference
    phot_table['mjd'] = mjd

    bins = np.arange(hp.nside2npix(nside)+1)-0.5
    result, be, bn = binned_statistic(catalog_hp[good_transform][detected], mag_difference, bins=bins)

    if return_table:
        return phot_table, result
    else:
        return result

