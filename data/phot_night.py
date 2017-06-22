import numpy as np
from lsst.all_sky_phot import readcr2
import glob
from photutils import Background2D, SigmaClip, MedianBackground, DAOStarFinder, CircularAperture, aperture_photometry, CircularAnnulus
from astropy.stats import sigma_clipped_stats
import matplotlib.pylab as plt
import sys

# Let's try running photometry on a night


def phot_files(files, phot_params=None, savefile='phot_night.npz', clip_negative=True):

    if phot_params is None:
        phot_params = {'background_size': 50, 'bk_clip_sigma': 3, 'bk_iter': 10,
                       'bk_filter_size': 6, 'dao_fwhm': 3.0, 'dao_thresh': 5.,
                       'apper_r': 5., 'ann_r_in': 6., 'ann_r_out': 8.,
                       'stat_region': [2000, 3000, 2000, 3000]}

    phot_tables = []

    maxi = float(np.size(files))
    for i, filename in enumerate(files):
        print 'reading image'
        im, header = readcr2(filename)
        sum_image = np.sum(im, axis=2).astype(float)

        # Do background subtraction
        print 'background'
        sigma_clip = SigmaClip(sigma=phot_params['bk_clip_sigma'], iters=phot_params['bk_iter'])
        bkg_estimator = MedianBackground()
        bkg = Background2D(sum_image, (phot_params['background_size'], phot_params['background_size']),
                           filter_size=(phot_params['bk_filter_size'], phot_params['bk_filter_size']),
                           sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
        bk_img = sum_image - bkg.background
        mean, median, std = sigma_clipped_stats(bk_img[phot_params['stat_region'][0]:phot_params['stat_region'][1],
                                                phot_params['stat_region'][2]:phot_params['stat_region'][3]])

        # Find sources
        print 'finding sources'
        daofind = DAOStarFinder(fwhm=phot_params['dao_fwhm'],
                                threshold=phot_params['dao_thresh']*std)
        sources = daofind(bk_img)

        # Set star and annulus positions
        positions = list(zip(sources['xcentroid'].data, sources['ycentroid'].data))
        apertures = CircularAperture(positions, r=phot_params['apper_r'])
        annulus_apertures = CircularAnnulus(positions, r_in=phot_params['ann_r_in'],
                                            r_out=phot_params['ann_r_out'])

        # I'm kind of sad that I can't do median to get the local background?
        print 'doing photometry'
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

        # Fill in the MJD
        phot_table['mjd'] = header['mjd']
        phot_tables.append(phot_table)

        # Progress bar
        progress = i/maxi*100
        text = "\rprogress = %.2f%%" % progress
        sys.stdout.write(text)
        sys.stdout.flush()

    if savefile is not None:
        np.savez(savefile, phot_tables=phot_tables)
    return phot_tables

if __name__ == '__main__':

    files = glob.glob('ut012616/*.cr2')
    phot_files(files, savefile='012616_night_phot.npz')
    files = glob.glob('ut012716/*.cr2')
    phot_files(files, savefile='012716_night_phot.npz')

