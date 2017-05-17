import numpy as np
from lsst.all_sky_phot import readcr2
import glob
#import pyds9
from photutils import Background2D, SigmaClip, MedianBackground, DAOStarFinder, CircularAperture, aperture_photometry, CircularAnnulus
from astropy.stats import sigma_clipped_stats
import matplotlib.pylab as plt
import sys

# Let's try running photometry on a night

files = glob.glob('ut012716/*.cr2')

phot_tables = []

maxi = float(np.size(files))
for i, filename in enumerate(files):
    im, header = readcr2(filename)
    sum_image = np.sum(im, axis=2).astype(float)

    # Do background subtraction
    sigma_clip = SigmaClip(sigma=3., iters=10)
    bkg_estimator = MedianBackground()
    bkg = Background2D(sum_image, (50, 50), filter_size=(6, 6),
                       sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
    bk_img = sum_image - bkg.background
    mean, median, std = sigma_clipped_stats(bk_img[2000:3000,2000:3000])
    daofind = DAOStarFinder(fwhm=3.0, threshold=5.*std)
    sources = daofind(bk_img)

    positions = list(zip(sources['xcentroid'].data, sources['ycentroid'].data))
    apertures = CircularAperture(positions, r=5.)
    annulus_apertures = CircularAnnulus(positions, r_in=6., r_out=8.)


    # I'm kind of sad that I can't do median to get the local background.
    apers = [apertures, annulus_apertures]
    phot_table = aperture_photometry(bk_img, apers)
    bkg_mean = phot_table['aperture_sum_1'] / annulus_apertures.area()
    bkg_sum = bkg_mean * apertures.area()
    final_sum = phot_table['aperture_sum_0'] - bkg_sum
    phot_table['residual_aperture_sum'] = final_sum

    phot_table['mjd'] = header['mjd']
    phot_tables.append(phot_table)
    progress = i/maxi
    text = "\rprogress = %.if%%" % progress
    sys.stdout.write(text)
    sys.stdout.flush()

np.savez('full_night.npz', phot_tables=phot_tables)
