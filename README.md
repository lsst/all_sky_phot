# all_sky_phot
Tools for converting all sky camera images into transparency maps.


## Dependencies
numpy

dcraw (for reading Canon cr2 files)

astropy

photutils (pip-installable, conda install -c astropy photutils)

astroquery (pip-installable, or conda install -c astropy astroquery)

## General Usage

The code includes a WCS class for mapping highly distorted all-sky images. The Yale Bright Star Catalog is included as a good source for bootstrapping a wcs fit.

Once a WCS solution for a camera has been fit, the `forced_phot` function can take an all-sky image along with it's WCS and a catalog of known stars to generate a transparency map.



