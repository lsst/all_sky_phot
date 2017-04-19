import numpy as np
import astropy
from astropy.modeling import models, fitting
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, ICRS
from astropy.modeling.projections import Pix2Sky_ZEA, Sky2Pix_ZEA, AffineTransformation2D



# I want to make a rough estimate of the coordinate solution, then I can use a night of data to make a 
# dense map of points?

# Zenith distance = 90 - alt
# set zenith x, zenith y, carmera rot angle
# pixAz = ax - camera_rot_angle
# dx = 1-sin(Pixax), dy = cos(Pixaz)
# x = ZenithX+dx, y=zenithY+dy


# Maybe use http://docs.astropy.org/en/stable/wcs/ to figure out how to write a ZPN or ZEA header?

# Seems like I could make a wcs, then do w.wcs_world2pix() on my stellar catalog. 

# More good stuff here: http://www2.mpia-hd.mpg.de/~robitaille/astropy/1.1/modeling/index.html

# If I identify a few stars, then I could use the astropy model fitting to get an initial solution, then iterate. 
# Not sure how I add deviations?

# Can use astropy to do alt az conversion: http://docs.astropy.org/en/stable/generated/examples/coordinates/plot_obs-planning.html#sphx-glr-generated-examples-coordinates-plot-obs-planning-py

# values from lsst.sims.utils.Site, but trying to avoid stack dependency
lsst_location = EarthLocation(lat=-30.2444*u.degree, lon=-70.7494*u.degree, height=2650.0*u.meter)

coords = ['05h55m10.30536 +07d24m25.4304']
beatlguese = SkyCoord(coords, ICRS, unit=(u.deg, u.hourangle))

time = Time(54000.5, format='mjd')

beAltAz = beatlguese.transform_to(AltAz(obstime=time, location=lsst_location))

# Need to generate alt,az coords for the stars in each frame. Then I can fit all the times simultaneously. 

# Eventually going to need to be able to predict the expected mags for the stars. 

# How do I deal with the different color pixels being spatially different? If the center falls on a green vs red pixel, that'll change things.

# for the astrometry. Looks like I need to daisy-chain some functions together? If I zenith project, then affine transform, then some distortions)?), that should do it?

# This makes a compound transformation!
proj_aff = Sky2Pix_ZEA + AffineTransformation2D

# I think I need to make a modeling sub-class that then takes u'x', u'y' and u'x_meas', u'y_meas', and outputs some statistic on radius difference.
# R_sq = (x-x_meas)**2 + (y-y_meas)**2. Oh, I could just make x_meas, y_meas set on __init__, then inputs x,y and outputs R**2, and make my expected R**2 = 0.
# do I just sum that? sum the sqrt? OR do I do that in the fitter? 

# Then, how do I put in distortions?  