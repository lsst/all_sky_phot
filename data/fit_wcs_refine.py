import numpy as np
from astropy.modeling.projections import Pix2Sky_ZEA, Sky2Pix_ZEA, AffineTransformation2D
from lsst.all_sky_phot import readcr2
from photutils import Background2D, SigmaClip, MedianBackground, DAOStarFinder, CircularAperture, aperture_photometry, CircularAnnulus
from astropy.stats import sigma_clipped_stats
import matplotlib.pylab as plt

# Given an initial guess, let's refine the wcs fit

def phot_all_sky_image(filename, channel=None, fwhm=3., background_window=50,
                       filter_size=6, phot_r=5., r_in=6., r_out=8., threshold=10.):

    im, header = readcr2(filename)
    if channel is None:
        im = np.sum(im, axis=2).astype(float)
    else:
        im = im[:, :, channel]

    # Do background subtraction
    sigma_clip = SigmaClip(sigma=3., iters=10)
    bkg_estimator = MedianBackground()
    bkg = Background2D(im, (background_window, background_window),
                       filter_size=(filter_size, filter_size),
                       sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
    bk_img = im - bkg.background
    mean, median, std = sigma_clipped_stats(bk_img[2000:3000, 2000:3000])
    daofind = DAOStarFinder(fwhm=fwhm, threshold=threshold*std)
    sources = daofind(bk_img)

    positions = list(zip(sources['xcentroid'].data, sources['ycentroid'].data))
    apertures = CircularAperture(positions, r=phot_r)
    annulus_apertures = CircularAnnulus(positions, r_in=r_in, r_out=r_out)

    # I'm kind of sad that I can't do median to get the local background.
    apers = [apertures, annulus_apertures]
    phot_table = aperture_photometry(bk_img, apers)
    bkg_mean = phot_table['aperture_sum_1'] / annulus_apertures.area()
    bkg_sum = bkg_mean * apertures.area()
    final_sum = phot_table['aperture_sum_0'] - bkg_sum
    phot_table['residual_aperture_sum'] = final_sum
    phot_table['mjd'] = header['mjd']
    return phot_table


# Initial affine transform guess
init_affine = [2.87356521e+03, 1.98559534e+03, -2.77295650e+01, -2.12539224e+00,
               -2.27577556e+00, 2.76348358e+01]

test_file = 'ut012716/ut012716.0106.long.cr2'
phot_table = phot_all_sky_image(test_file)

aff = AffineTransformation2D()
aff.translation.value = init_affine[0:2]
aff.matrix.value = init_affine[2:]

x, y = aff.inverse(phot_table['xcenter'], phot_table['ycenter'])
tosky = Pix2Sky_ZEA()
az, alt = tosky(x, y)

