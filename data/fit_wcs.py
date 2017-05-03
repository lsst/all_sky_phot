from __future__ import print_function
import numpy as np
from astroquery.simbad import Simbad
import matplotlib.pylab as plt
import astropy
from astropy.modeling import models, fitting
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, ICRS, Longitude, Latitude
from astropy.modeling.projections import Pix2Sky_ZEA, Sky2Pix_ZEA, AffineTransformation2D, Sky2Pix_AZP
from scipy.optimize import minimize
from astropy.wcs import Sip
from astropy.modeling.polynomial import Legendre2D

lsst_location = EarthLocation(lat=-30.2444*u.degree, lon=-70.7494*u.degree, height=2650.0*u.meter)


# Let's take a few stars from a few frames and try to fit a WCS projection. Then we can try to transform more.

names = ['star_name', 'x', 'y', 'mjd']
types = ['|S10', float, float, float]

obs_stars = np.loadtxt('starcoords.dat', dtype=list(zip(names, types)), skiprows=1, delimiter=',')

names.extend(['alt', 'az'])
types.extend([float, float])

new_cols = np.zeros(obs_stars.size, dtype=list(zip(names, types)))
for name in obs_stars.dtype.names:
    new_cols[name] = obs_stars[name]

obs_stars = new_cols

ustars = np.unique(obs_stars['star_name'])
names = ['star_name', 'ra', 'dec']
types = ['|S20']*3
star_coords = np.zeros(ustars.size, dtype=list(zip(names, types)))

# Get the RA,Dec values for each star from Simbad
for i, star_name in enumerate(ustars):
    result = Simbad.query_object(star_name)
    star_coords[i]['star_name'] = star_name
    ra = result['RA'][0].split(' ')
    ra = ra[0]+'h'+ra[1]+'m'+ra[2]
    star_coords[i]['ra'] = ra
    dec = result['DEC'][0].split(' ')
    dec = dec[0]+'d'+dec[1]+'m'+dec[2]
    star_coords[i]['dec'] = dec


# Predict the alt, az values for each star at each MJD
ra = Longitude(star_coords['ra'], unit=u.hourangle)
dec = Latitude(star_coords['dec'], unit=u.deg)
star_coords = SkyCoord(ra=ra, dec=dec, frame=ICRS)

utimes = np.unique(obs_stars['mjd'])
star_names = []
mjds = []
alts = []
azs = []
for time in utimes:
    time_mjd = Time(time, format='mjd')
    trans = star_coords.transform_to(AltAz(obstime=time_mjd, location=lsst_location))
    star_names.extend(ustars.tolist())
    mjds.extend([time]*ustars.size)
    alts.extend(trans.alt.value.tolist())
    azs.extend(trans.az.value.tolist())


names = ['star_name', 'alt', 'az', 'mjd']
types = ['|S10', float, float, float]

predicted_array = np.zeros(len(azs), dtype=list(zip(names, types)))
predicted_array['star_name'] = np.array(star_names)
predicted_array['alt'] = np.array(alts)
predicted_array['az'] = np.array(azs)
predicted_array['mjd'] = np.array(mjds)

predicted_array.sort(order=['star_name', 'mjd'])
obs_stars.sort(order=['star_name', 'mjd'])
hash1 = np.core.defchararray.add(obs_stars['star_name'], obs_stars['mjd'].astype('|S20'))
hash2 = np.core.defchararray.add(predicted_array['star_name'], predicted_array['mjd'].astype('|S20'))

good = np.in1d(hash2, hash1)
predicted_array = predicted_array[good]
obs_stars['alt'] = predicted_array['alt']
obs_stars['az'] = predicted_array['az']

# Fit the projection!
# Now we have x,y values matched up to the expected alt,az values


class AZP_affine(object):
    def __init__(self, x, y, alt, az):
        self.projection = Sky2Pix_AZP()
        self.affine = AffineTransformation2D()
        self.x = x
        self.y = y
        self.alt = alt
        self.az = az

    def compute(self, x0):
        self.affine.translation.value = x0[0:2]
        self.affine.matrix.value = x0[2:6]
        self.projection.mu = x0[6]
        self.projection.gamma = x0[7]
        newx, newy = self.projection(self.az, self.alt)
        newx, newy = self.affine(newx, newy)
        return newx, newy

    def __call__(self, x0):
        newx, newy = self.compute(x0)
        residual = np.sum((newx - self.x)**2 + (newy-self.y)**2)
        return residual


class ZEA_affine(object):
    def __init__(self, x, y, alt, az):
        self.projection = Sky2Pix_ZEA()
        self.affine = AffineTransformation2D()
        self.x = x
        self.y = y
        self.projx, self.projy = self.projection(az, alt)

    def compute(self, x0):
        self.affine.translation.value = x0[0:2]
        self.affine.matrix.value = x0[2:]
        newx, newy = self.affine(self.projx, self.projy)
        return newx, newy

    def __call__(self, x0):
        """
        x0 : numpy.array
            x-trans, ytrans, affine[0,0], [0,1], [1,0], [1,1]
        """
        newx, newy = self.compute(x0)

        residual = np.sum((newx - self.x)**2 + (newy-self.y)**2)
        return residual


def arr2abapbp(x, a_order, b_order, ap_order, bp_order):
    a_size = (a_order+1)**2
    b_size = (b_order+1)**2
    ap_size = (ap_order+1)**2
    bp_size = (bp_order+1)**2
    a_ind = np.arange(a_size)
    b_ind = np.arange(b_size) + a_ind.max() + 1
    ap_ind = np.arange(ap_size) + b_ind.max() + 1
    bp_ind = np.arange(bp_size) + ap_ind.max() + 1

    a = x[a_ind].reshape((a_order + 1, a_order + 1))
    b = x[b_ind].reshape((b_order + 1, b_order + 1))
    ap = x[ap_ind].reshape((ap_order + 1, ap_order + 1))
    bp = x[bp_ind].reshape((bp_order + 1, bp_order + 1))

    return a, b, ap, bp


class AZP_SIP(object):
    def __init__(self, x, y, alt, az, ap_order=2, bp_order=2):
        self.projection = Sky2Pix_AZP()
        self.affine = AffineTransformation2D()
        self.sip = Sip
        self.x = x
        self.y = y
        self.alt = alt
        self.az = az
        self.ap_order = ap_order
        self.bp_order = bp_order
        self.ap_size = (self.ap_order+1)**2
        self.bp_size = (self.bp_order+1)**2
        self.ap_ind = np.arange(self.ap_size)
        self.bp_ind = np.arange(self.bp_size) + self.ap_ind.max()+1
        self.affine_ind = np.arange(6) + self.bp_ind.max() + 1
        self.crpix = np.zeros(2)
        self.zs = np.zeros((3, 3))

    def compute(self, x0):
        """
        OK, I think I project from sky to fov, then SIP foc2pix, then affine to
        shift, rotate, scale the pixel coords.
        """

        #a = x0[self.a_ind].reshape((self.a_order + 1, self.a_order + 1))
        #b = x0[self.b_ind].reshape((self.b_order + 1, self.b_order + 1))
        ap = x0[self.ap_ind].reshape((self.ap_order + 1, self.ap_order + 1))
        bp = x0[self.bp_ind].reshape((self.bp_order + 1, self.bp_order + 1))
        #a, b, ap, bp = arr2abapbp(x0, self.a_order, self.b_order, self.ap_order, self.bp_order)
        sip = self.sip(self.zs, self.zs, ap, bp, self.crpix)

        # Project to plane
        self.projection.mu = x0[-2]
        self.projection.gamma = x0[-1]
        newx, newy = self.projection(self.az, self.alt)

        # Apply SIP distorions
        new_coords = sip.foc2pix(np.vstack((newx, newy)).T, 0)
        newx = new_coords[:, 0]
        newy = new_coords[:, 1]

        # Apply affine transformation
        self.affine.translation.value = x0[self.affine_ind[0:2]]
        self.affine.matrix.value = x0[self.affine_ind[2:]]
        newx, newy = self.affine(newx, newy)

        return newx, newy

    def __call__(self, x0):
        newx, newy = self.compute(x0)
        residual = np.sum((newx - self.x)**2 + (newy-self.y)**2)
        return residual


class AZP_Legendre(object):
    """
    Use the AZP projection, and then affine and legnedre polynomials
    """
    def __init__(self, x, y, alt, az, leg_x_order=2, leg_y_order=2):
        self.projection = Sky2Pix_AZP()
        self.affine = AffineTransformation2D()
        self.x = x
        self.y = y
        self.alt = alt
        self.az = az
        self.x_leg = Legendre2D(leg_x_order, leg_y_order)
        self.y_leg = Legendre2D(leg_x_order, leg_y_order)

    def compute(self, x0):
        self.affine.translation.value = x0[0:2]
        self.affine.matrix.value = x0[2:6]
        self.projection.mu = x0[6]
        self.projection.gamma = x0[7]
        self.x_leg.parameters = x0[8:blah]
        #self.y_leg.parameters = x0[]

        # Project to x-y. 
        newx, newy = self.projection(self.az, self.alt)
        # Offset in polar coordinates
        r = (newx**2+newy**2)**0.5
        theta = np.arctan2(newy, newx)
        xoff = self.x_leg(r, theta)
        yoff = self.y_leg(r, theta)


        # rotate, shift, and scale result
        newx, newy = self.affine(newx, newy)
        return newx, newy

    def __call__(self, x0):
        newx, newy = self.compute(x0)
        residual = np.sum((newx - self.x)**2 + (newy-self.y)**2)
        return residual


def plot_projection(obs_stars, fitx, fity, filename=None):
    
    """
    fig, ax = plt.subplots(1, 1)
    ack = ax.scatter(obs_stars['x'], obs_stars['y'], c=obs_stars['alt'], s=50)
    ax.set_xlabel('x (pix)')
    ax.set_ylabel('y (pix)')
    cb = fig.colorbar(ack)
    cb.set_label('altitude (deg)')
    #fig.savefig('Plots/first_wcs_fit.png')
    """

    fig, (ax1,ax2) = plt.subplots(2)
    ack = ax1.scatter(obs_stars['x'], obs_stars['y'], c=fitx-obs_stars['x'], s=50)
    ax1.set_xlabel('x (pix)')
    ax1.set_ylabel('y (pix)')
    ax1.set_title('Residuals')
    cb = fig.colorbar(ack, ax=ax1)
    cb.set_label(r'$\Delta x$ (pix)')

    ack = ax2.scatter(obs_stars['x'], obs_stars['y'], c=fity-obs_stars['y'], s=50)
    ax2.set_xlabel('x (pix)')
    ax2.set_ylabel('y (pix)')
    ax2.set_title('Residuals')
    cb = fig.colorbar(ack, ax=ax2)
    cb.set_label(r'$\Delta y$ (pix)')


    if filename is not None:
        fig.savefig(filename)



# hmm, not clear how to use the astropy fitting stuff here... just revert to regular scipy optimize
fun = ZEA_affine(obs_stars['x'], obs_stars['y'], obs_stars['alt'], obs_stars['az'])
x0 = np.array([np.median(obs_stars['x']), np.median(obs_stars['y']), 1., 0., 0., 1.])
fit_result = minimize(fun, x0)

fitx, fity = fun.compute(fit_result.x)

xresid = (fitx-obs_stars['x']).std()
yresid = (fity-obs_stars['y']).std()
print('Fitted parameters', fit_result.x)
print('ZEA_affine fit result = %f' % fun(fit_result.x))

plot_projection(obs_stars, fitx, fity, filename='Plots/resids_wcs_fit.png')
x0_new = np.zeros(8)
x0_new[0:6] += fit_result.x
fun = AZP_affine(obs_stars['x'], obs_stars['y'], obs_stars['alt'], obs_stars['az'])

fit_result = minimize(fun, x0_new)
fitx, fity = fun.compute(fit_result.x)
plot_projection(obs_stars, fitx, fity)
print('AZP_affine fit result = %f' % fun(fit_result.x))


a_order = 2
b_order = 2
x0_new = np.zeros((a_order+1)**2+(b_order+1)**2 + 8)
x0_new[-8:] = fit_result.x[-8:]

# huh, the SIP has seperate terms for foc2pix and pix2foc...Seems like there should be 
# method for keeping those in sync...

fun = AZP_SIP(obs_stars['x'], obs_stars['y'], obs_stars['alt'], obs_stars['az'],
              ap_order=a_order, bp_order=b_order)

fit_result = minimize(fun, x0_new)
fitx, fity = fun.compute(fit_result.x)
plot_projection(obs_stars, fitx, fity)
print('AZP_SIP order %i fit result = %f' % (a_order, fun(fit_result.x)))



