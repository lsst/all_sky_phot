import numpy as np
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, ICRS
import astropy.units as u
from astropy.time import Time

# Some handy utilities for bootstrapping WCS solutions


lsst_location = EarthLocation(lat=-30.2444*u.degree,
                              lon=-70.7494*u.degree,
                              height=2650.0*u.meter)


def robustRMS(x):
    """
    RMS based on the inter-quartile range.
    """
    iqr = np.percentile(x, 75)-np.percentile(x, 25)
    rms = iqr/1.349  # approximation
    return rms


def radec2altaz(ra, dec, mjd, location=lsst_location):
    """I need a stupid converter
    """
    time = Time(mjd, format='mjd')
    coords = SkyCoord(ra=ra*u.degree, dec=dec*u.degree)
    ack = coords.transform_to(AltAz(obstime=time, location=location))
    return ack.alt.value, ack.az.value


def apply_wcs_to_photometry(ptable, w, location=lsst_location, zp=None):
    """
    Take a photometry table and add ra and dec cols

    Parameters
    ----------
    ptable : astropy table
        Needs columns xcenter, ycenter, and mjd.
        Assumes all the mjd are the same
    wcs : wcs object
        the World Coordinate System object
    location : astropy EarthLocation object

    Returns
    -------
    Photometry table with columns added for the alt, az, ra, dec
    """
    time = Time(ptable['mjd'].max(), format='mjd')
    az, alt = w.all_pix2world(ptable['xcenter'], ptable['ycenter'], 0)
    coords = AltAz(az=az*u.degree, alt=alt*u.degree, location=location, obstime=time)
    sky_coords = coords.transform_to(ICRS)
    ptable['alt_wcs'] = coords.alt
    ptable['az_wcs'] = coords.az
    ptable['ra_wcs'] = sky_coords.ra
    ptable['dec_wcs'] = sky_coords.dec

    if zp is not None:
        ptable['mag'] = -2.5*np.log10(ptable['residual_aperture_sum'].data) - zp

    return ptable


def match_catalog(ptable, catalog, cat_mags, location=lsst_location):
    """Match the photometry to a catalog

    Add matched_alt, az, ra, dec columns, and distance to match. Maybe also matched star ID
    """

    good_coords = np.where((~np.isnan(ptable['ra_wcs'])) & (~np.isnan(ptable['dec_wcs'])))
    phot_cat = SkyCoord(ra=ptable['ra_wcs'].value[good_coords]*u.degree,
                        dec=ptable['dec_wcs'].value[good_coords]*u.degree)
    idx, d2d, d3d = phot_cat.match_to_catalog_sky(catalog)

    # Clear any old columns
    ptable['ra_matched'] = -666.
    ptable['dec_matched'] = -666.
    ptable['alt_matched'] = -666.
    ptable['az_matched'] = -666.
    ptable['d2d'] = -666.
    ptable['matched_Vmag'] = -666.
    ptable['bright_star_idx'] = -666
    ptable['d2d'][good_coords] = d2d
    ptable['matched_Vmag'][good_coords] = cat_mags[idx]
    ptable['bright_star_idx'][good_coords] = idx

    ptable['ra_matched'][good_coords] = catalog.ra[idx]
    ptable['dec_matched'][good_coords] = catalog.dec[idx]

    time = Time(ptable['mjd'].max(), format='mjd')
    ack = catalog.transform_to(AltAz(obstime=time, location=location))
    ptable['alt_matched'][good_coords] = ack.alt[idx]
    ptable['az_matched'][good_coords] = ack.az[idx]
    ptable

    return ptable


def trim_stars(photo_array, d2d_limit=1., mag_limit=1.5, nstars=5):
    """Take an array that has been matched, and try to filter out the spurrious 
    matches to make a reliable x,y,alt,az table for fitting.

    Return the indices that meet the criteria
    """

    # Maybe for each star, make a running median and robust rms for x,y,mag as 
    # a function of time. Then reject outliers. 
    # Array to save if we think a point is a good match
    good_match = np.zeros(np.size(photo_array), dtype=bool)
    meet_criteria = np.where((photo_array['d2d'] < d2d_limit) &
                             (np.abs(photo_array['mag'] - photo_array['matched_Vmag']) < mag_limit))
    good_match[meet_criteria] = True

    # The unique stars that I should loop though
    ustars = np.unique(photo_array['bright_star_idx'][good_match])
    print 'found %i unque stars' % np.size(ustars)
    for star in ustars:
        matching = np.where((photo_array['bright_star_idx'] == star) & (good_match == True))
        # toss if we don't have enough of this star
        if np.size(matching[0]) < nstars:
            good_match[matching] = False
        else:
            if np.size(np.unique(photo_array['mjd'][matching])) < np.size(matching[0]):
                # Need to week out times where we have two matches
                for mjd in photo_array['mjd'][matching]:
                    # Need to pick the best of these
                    indxs = np.where((photo_array['mjd'] == mjd) &
                                     (photo_array['bright_star_idx'] == star) &
                                     (good_match == True))[0]
                    # Just take the closest one. Maybe consider mag in the future
                    good_match[indxs[np.where(photo_array['d2d'][indxs] >
                                              photo_array['d2d'][indxs].min())]] = False
            matching = np.where((photo_array['bright_star_idx'] == star) & (good_match == True))
            # If we don't match the star enough times, toss it.
            if np.size(matching[0]) < nstars:
                good_match[matching] = False
    return good_match






