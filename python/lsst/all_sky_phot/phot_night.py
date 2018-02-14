from __future__ import print_function
import numpy as np
from lsst.all_sky_phot import readcr2
from photutils import Background2D, MedianBackground, DAOStarFinder, CircularAperture, aperture_photometry, CircularAnnulus
from astropy.stats import sigma_clipped_stats, SigmaClip
import sys


__all__ = ['phot_night', 'default_phot_params', 'phot_image']


def default_phot_params():
    """Make some handy default photometry paramters

    Parameters
    ----------
    background_size : int (50)
        Used by photutils.Background2D. box_size used for background calc.
    bk_clip_sigma : float (3)
        Sigma clipping used by photutils.Background2D.
    bkiter : int (10)
        Number of iterations used in sigma clipping.
    bk_filter_size : int (6)
        Window size of the 2D median filter used on background image by photutils.Background2D.
    dao_fwhm : float (3.)
        FWHM used by photutils.DAOStarFinder.
    dao_thresh : float (5.)
        Number of standard deviation threshold used by photutils.DAOStarFinder.
    appr_r : float (5.)
        Radius used by photutils.CircularAperture.
    ann_r_in : float (6.)
        Inner radial radius used by photutils.CircularAnnulus.
    ann_r_out : float(8.)
        Outer radial radial used by photutils.CircularAnnulus.
    stat_region : list ([2000, 3000, 2000, 3000])
        Pixel coordinates defining the region on the background subtracted image where stats (e.g., std)
        are computed.
    """

    phot_params = {'background_size': 50, 'bk_clip_sigma': 3, 'bk_iter': 10,
                   'bk_filter_size': 6, 'dao_fwhm': 3.0, 'dao_thresh': 5.,
                   'apper_r': 5., 'ann_r_in': 6., 'ann_r_out': 8.,
                   'stat_region': [2000, 3000, 2000, 3000]}
    return phot_params


def phot_image(image, phot_params=None, clip_negative=True, verbose=False):
    """Run detection and photometry on a single image

    Parameters
    ----------
    image : array
        The image to perform photometry on
    phot_params : dict (None)
        Dictionary holding common photometry kwargs. Loads defaults from default_phot_params
        if None.
    clip_negative : bool (True)
        Remove any sources that have negative flux.
    verbose : bool (False)
        Print out steps.
    """
    # Do background subtraction
    if verbose:
        print('background')
    sigma_clip = SigmaClip(sigma=phot_params['bk_clip_sigma'], iters=phot_params['bk_iter'])
    bkg_estimator = MedianBackground()
    bkg = Background2D(image, (phot_params['background_size'], phot_params['background_size']),
                       filter_size=(phot_params['bk_filter_size'], phot_params['bk_filter_size']),
                       sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
    bk_img = image - bkg.background
    mean, median, std = sigma_clipped_stats(bk_img[phot_params['stat_region'][0]:phot_params['stat_region'][1],
                                            phot_params['stat_region'][2]:phot_params['stat_region'][3]])

    # Find sources
    if verbose:
        print('finding sources')
    daofind = DAOStarFinder(fwhm=phot_params['dao_fwhm'],
                            threshold=phot_params['dao_thresh']*std)
    sources = daofind(bk_img)

    # Set star and annulus positions
    positions = list(zip(sources['xcentroid'].data, sources['ycentroid'].data))
    apertures = CircularAperture(positions, r=phot_params['apper_r'])
    annulus_apertures = CircularAnnulus(positions, r_in=phot_params['ann_r_in'],
                                        r_out=phot_params['ann_r_out'])

    # I'm kind of sad that I can't do median to get the local background?
    if verbose:
        print('doing photometry')
    apers = [apertures, annulus_apertures]
    phot_table = aperture_photometry(bk_img, apers)
    bkg_mean = phot_table['aperture_sum_1'] / annulus_apertures.area()
    bkg_sum = bkg_mean * apertures.area()
    final_sum = phot_table['aperture_sum_0'] - bkg_sum
    phot_table['residual_aperture_sum'] = final_sum

    # Clip any negative flux objects
    if clip_negative:
        good = np.where(phot_table['residual_aperture_sum'] > 0)
        phot_table = phot_table[good]

    return phot_table


def phot_night(files, phot_params=None, savefile='phot_night.npz', clip_negative=True,
               verbose=False, progress_bar=True):
    """Run apperture photometry on a list of files

    Parameters
    ----------
    files : list of str
        Filenames of cr2 images to read
    phot_params : dict (None)
        Dictionary holding common photometry kwargs. Loads defaults from default_phot_params
        if None.
    savefile : str ('phot_night.npz')
        The numpy npz file to save the resulting photometry tables to.
    clip_negative : bool (True)
        Should stars that end up with negative flux after background subtraction be clipped.
    verbose : bool (False)
        Print out progress for each frame
    progress_bar : bool (True)
        Print out a progress bar for how many files have been completed
    """

    if phot_params is None:
        phot_params = default_phot_params()

    phot_tables = []

    maxi = float(np.size(files))
    for i, filename in enumerate(files):
        if verbose:
            print('reading image')
        im, header = readcr2(filename)
        sum_image = np.sum(im, axis=2).astype(float)

        phot_table = phot_image(sum_image, phot_params=phot_params, clip_negative=clip_negative,
                                verbose=verbose)

        # Fill in the MJD
        phot_table['mjd'] = header['mjd']
        phot_tables.append(phot_table)

        # Progress bar
        if progress_bar:
            progress = i/maxi*100
            text = "\rprogress = %.2f%%" % progress
            sys.stdout.write(text)
            sys.stdout.flush()

    if savefile is not None:
        np.savez(savefile, phot_tables=phot_tables)
    return phot_tables
