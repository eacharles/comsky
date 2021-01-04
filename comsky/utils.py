"""Utility functions"""

import numpy as np
import healpy as hp
import scipy.stats as sps


def SafeRVS(dist, nevt):
    """Sample a number of points from a distribution

    Parameters
    ----------
    dist : `scipy.stats.rv_continuous` or `float`
        Distribution to sample from, or float
    nevt : `int`
        Number of points to sample

    Returns
    -------
    vals : array_like
        Sampled values

    Notes
    -----
    if dist is a float, this will just return an array of size `nevt` all set to `dist`
    """
    if np.isscalar(dist): #pragma: no cover
        return np.ones(nevt)*dist
    return dist.rvs(size=nevt)



def AddRingToMap(m, pix, nside, radius=30, width=5):
    """Given a healpix map and a sky direction (given here as the healpix pixel index),
    add the back-projected compton cone around that pixel to the map.
    That is done by finding the pixels in a ring (with given radius and width)
    around the given direction,
    and adding 1 to the map value of the pixels in that ring.
    (Might want to normalize it by the ring area eventually.)

    Parameters
    ----------
    m : `healpy map`
       A `healpy` map in RING format
    pix : `int`
       HEALpix index of central direction
    nside : `int`
       HEALpix nside parameter
    radius : `float`
       Compton ring radius, in degrees
    width : `float`
       Compton ring width, in degrees
    """
    rO=np.deg2rad(radius+0.5*width)
    rI=np.deg2rad(radius-0.5*width)
    vec=hp.pixelfunc.pix2vec(nside, pix)
    sel = hp.query_disc(nside, vec, rO)
    m[sel] += 1.
    sel = hp.query_disc(nside, vec, rI)
    m[sel] -= 1.



def GetNptsForResolution(nside, alpha=1., factor=2.):
    """Estimate the number of points needed to sample a healpix map with a ring of radius alpha

    Parameters
    ----------
    nside : `int`
        HEALPix nside
    alpha : `float`
        Ring radius (in radians)
    factor : `float`
        sampling factor

    Returns
    -------
    npts : int
        Number of points use to sample ring
    """
    return np.ceil(factor*2*np.pi*alpha / hp.pixelfunc.nside2resol(nside)).astype(int)


def GetLociPoints(lon, colat, alpha, phi):
    """Get points around a ring

    Parameters
    ----------
    lon : float or array_like
        Longitude of center of ring (phi, in radians)
    colat : float or array_like
        Co-latitude of center of ring (theta, in radians)
    alpha : float or array_like
        Radius of ring (in radians)
    phi : float or array_like
        Parameter that various along ring

    Returns
    -------
    x,y,z : components of points around ring

    Notes
    -----
    If any of lon, colat and alpha are array_like, the phi should have the same shape
    this will effectively pick one point from each ring.
    If lon, colat and alpha are all scalar, then phi should be an array, and this will
    define points on a ring

    Uses equations define here https://math.stackexchange.com/questions/643130/circle-on-sphere
    """
    s_a = np.sin(alpha)
    s_b = np.sin(colat)
    s_c = np.sin(lon)

    c_a = np.cos(alpha)
    c_b = np.cos(colat)
    c_c = np.cos(lon)

    s_t = np.sin(phi)
    c_t = np.cos(phi)

    x_1 = s_a * c_b * c_c
    x_2 = s_a * s_c
    x_3 = -1 * c_a * s_b * c_c

    y_1 = -1 * s_a * c_b * s_c
    y_2 = s_a * c_c
    y_3 = c_a * s_b * s_c

    z_1 = s_a * s_b
    z_2 = c_a * c_b

    x = (x_1 * c_t) + (x_2 * s_t) + x_3
    y = (y_1 * c_t) + (y_2 * s_t) + y_3
    z = (z_1 * c_t) + z_2

    return np.vstack([x,y,z])


def GetLociRing(lon, colat, alpha, npts):
    """Get points around a ring

    Parameters
    ----------
    lon : float or array_like
        Longitude of center of ring (phi, in radians)
    colat : float or array_like
        Co-latitude of center of ring (theta, in radians)
    alpha : float or array_like
        Radius of ring (in radians)
    npts : int
        Number of points to use

    Returns
    -------
    x,y,z : components of points around ring
    """
    phi_vec = np.linspace(0, 2*np.pi, npts)
    return GetLociPoints(lon, colat, alpha, phi_vec)


def SamplePointsFromRings(lon, colat, alpha, phi_dist):
    """Sample a point from a bunch of rings

    Parameters
    ----------
    lon : array_like
        Longitude of center of ring (phi, in radians)
    colat : array_like
        Co-latitude of center of ring (theta, in radians)
    alpha : array_like
        Radius of ring (in radians)
    phi_dist : `scipy.stats.rv_continuous`
        Distribution to sample the ring phi parameter from

    Returns
    -------
    x,y,z : components of points around ring
    """

    npts = lon.size
    phi = phi_dist.rvs(npts)
    return GetLociPoints(lon, colat, alpha, phi)



def AddLociToMap(m, lon, colat, alpha, **kwargs):
    """Add points around a ring to a HEALPix map

    Parameters
    ----------
    m : array_like
        HEALPix map to fill
    lon : float
        Longitude of center of ring (phi, in radians)
    colat : float
        Co-latitude of center of ring (theta, in radians)

    Keywords
    --------
    alpha : float
        Radius of ring (in radians)
    factor : int
        sampling factor to define number of points per ring
    """
    factor = kwargs.get('factor', 2.)
    normed = kwargs.get('normed', False)

    nside = hp.pixelfunc.npix2nside(m.size)
    npts = GetNptsForResolution(nside, alpha=alpha, factor=factor)

    vecs = GetLociRing(lon, colat, alpha, npts)
    ipix = hp.pixelfunc.vec2pix(nside, vecs[0], vecs[1], vecs[2])

    weight = 1.
    if normed:
        weight /= float(npts)

    for ipix_ in ipix:
        m[ipix_] += weight


def AddSmearedEventsToMap(m, l, b, **kwargs):
    """
    Parameters
    ----------
    m : `array_like`
        HEALPix map
    l : `float`
        Source Galactic Longitude (in degrees)
    b : `float`
        Source Galactic Latitude (in degrees)

    Keywords
    --------
    alpha_true_dist : `scipy.stats.rv_continuous`
        Distribution of Compton ring size
    delta_alpha_dist : `scipy.stats.rv_continuous`
        Distribution of different between True and Observed Compton ring sizes
    phi_true_dist : `scipy.stats.rv_continuous`
        Distribution of Compton scattering angle
    """
    alpha_true_dist = kwargs['alpha_true_dist']
    delta_alpha_dist = kwargs['delta_alpha_dist']
    phi_true_dist = kwargs['phi_true_dist']

    nevt = l.size
    theta, phi = hp.pixelfunc.lonlat2thetaphi(l, b)
    alpha_true = SafeRVS(alpha_true_dist, nevt)
    delta_alpha = SafeRVS(delta_alpha_dist, nevt)
    phi_true = SafeRVS(phi_true_dist, nevt)
    center_vecs =  GetLociPoints(phi, theta, alpha_true, phi_true)
    alpha_obs = alpha_true + delta_alpha
    theta, phi = hp.pixelfunc.vec2ang(center_vecs.T)
    for theta_, phi_, alpha_ in zip(theta, phi, alpha_obs):
        AddLociToMap(m, phi_, theta_, alpha=alpha_, normed=True)


def AddSampledEventsToMap(m, l, b, **kwargs):
    """
    Parameters
    ----------
    m : `array_like`
        HEALPix map
    l : `float`
        Source Galactic Longitude (in degrees)
    b : `float`
        Source Galactic Latitude (in degrees)

    Keywords
    --------
    alpha_true_dist : `scipy.stats.rv_continuous`
        Distribution of Compton ring size
    delta_alpha_dist : `scipy.stats.rv_continuous`
        Distribution of different between True and Observed Compton ring sizes
    phi_true_dist : `scipy.stats.rv_continuous`
        Distribution of Compton scattering angle
    phi_sample_dist : `scipy.stats.rv_continuous`
        Distribution of Phi angles around ring
    """
    alpha_true_dist = kwargs['alpha_true_dist']
    delta_alpha_dist = kwargs['delta_alpha_dist']
    phi_true_dist = kwargs['phi_true_dist']
    phi_sample_dist = kwargs['phi_sample_dist']

    nevt = l.size
    nside = hp.pixelfunc.npix2nside(m.size)

    theta, phi = hp.pixelfunc.lonlat2thetaphi(l, b)
    alpha_true = SafeRVS(alpha_true_dist, nevt)
    delta_alpha = SafeRVS(delta_alpha_dist, nevt)
    phi_true = SafeRVS(phi_true_dist, nevt)
    phi_obs = SafeRVS(phi_sample_dist, nevt)
    center_vecs = GetLociPoints(phi, theta, alpha_true, phi_true)
    alpha_obs = alpha_true + delta_alpha
    theta_cent, phi_cent = hp.pixelfunc.vec2ang(center_vecs.T)
    vecs = GetLociPoints(phi_cent, theta_cent, alpha_obs, phi_obs)
    ipix = hp.pixelfunc.vec2pix(nside, vecs[0], vecs[1], vecs[2])
    for ipix_ in ipix:
        m[ipix_] += 1


def GetDefaultDistributions(radius=30., width=5.):
    """Make default distributions

    Parameters
    ----------
    radius : float
        Average Compton Ring radius (in degrees)
    width : float
        Width of distribution of Compton Ring Radii (in degrees)

    Returns
    -------
    alpha_true_dist : `scipy.stats.rv_continuous`
        Distribution of Compton ring size
    delta_alpha_dist : `scipy.stats.rv_continuous`
        Distribution of different between True and Observed Compton ring sizes
    phi_true_dist : `scipy.stats.rv_continuous`
        Distribution of Compton scattering angle
    """
    sigma = width/2.3548
    return dict(alpha_true_dist=sps.norm(loc=np.radians(radius), scale=np.radians(width)),
                    delta_alpha_dist=sps.norm(scale=np.radians(sigma)),
                    phi_true_dist=sps.uniform(scale=2*np.pi))



def MakePointSource(nPS, nside, l, b, **kwargs):
    """Simulate an observation of a point source

    Parameters
    ----------
    nPS : `int`
        Number of events to simulate
    nside : `int`
        HEALPix nside parameter
    l : `float`
        Source Galactic Longitude (in degrees)
    b : `float`
        Source Galactic Latitude (in degrees)

    Keywords
    --------
    radius : float
        Average Compton Ring radius (in degrees)
    width : float
        Width of distribution of Compton Ring Radii (in degrees)
    sample : bool
        If true, sample a single point from each ring

    Returns
    -------
    m : array_like
        HEALPix map filled with rings
    """
    radius=kwargs.get('radius', 30.)
    width=kwargs.get('width', 5.)
    sample=kwargs.get('sample', False)

    npix = hp.pixelfunc.nside2npix(nside)
    m = np.zeros(npix)

    l_vec = np.ones(nPS)*l
    b_vec = np.ones(nPS)*b

    dists = GetDefaultDistributions(radius, width)
    if sample:
        dists['phi_sample_dist'] = dists['phi_true_dist']
        AddSampledEventsToMap(m, l_vec, b_vec, **dists)
    else:
        AddSmearedEventsToMap(m, l_vec, b_vec, **dists)
    return m


def MakeIsotropicBackground(nIso, nside, **kwargs):
    """Simulate an observation of isotropic background

    Parameters
    ----------
    nIso : `int`
        Number of events to simulate
    nside : `int`
        HEALPix nside parameter

    Keywords
    --------
    radius : float
        Average Compton Ring radius (in degrees)
    width : float
        Width of distribution of Compton Ring Radii (in degrees)
    sample : bool
        If true, sample a single point from each ring

    Returns
    -------
    m : array_like
        HEALPix map filled with rings
    """
    radius=kwargs.get('radius', 30.)
    width=kwargs.get('width', 5.)
    sample=kwargs.get('sample', False)

    npix = hp.pixelfunc.nside2npix(nside)
    m= np.zeros(npix)

    ipix = sps.randint.rvs(0, npix-1, size=nIso)
    l_vec, b_vec = hp.pix2ang(nside, ipix, lonlat=True)

    dists = GetDefaultDistributions(radius, width)
    if sample:
        dists['phi_sample_dist'] = dists['phi_true_dist']
        AddSampledEventsToMap(m, l_vec, b_vec, **dists)
    else:
        AddSmearedEventsToMap(m, l_vec, b_vec, **dists)
    return m


def MakeGalacticBackground(nGal, nside, **kwargs):
    """Simulate an observation of galactic background

    Parameters
    ----------
    nGal : `int`
        Number of events to simulate
    nside : `int`
        HEALPix nside parameter

    Keywords
    --------
    radius : float
        Average Compton Ring radius (in degrees)
    width : float
        Width of distribution of Compton Ring Radii (in degrees)
    sample : bool
        If true, sample a single point from each ring

    Returns
    -------
    m : array_like
        HEALPix map filled with rings
    """
    radius=kwargs.get('radius', 30.)
    width=kwargs.get('width', 5.)
    sample=kwargs.get('sample', False)

    npix = hp.pixelfunc.nside2npix(nside)
    m = np.zeros(npix)

    l_vec = sps.uniform(loc=-180, scale=360).rvs(size=nGal)
    b_vec = sps.norm(scale=2.).rvs(size=nGal)
    dists = GetDefaultDistributions(radius, width)
    if sample:
        dists['phi_sample_dist'] = dists['phi_true_dist']
        AddSampledEventsToMap(m, l_vec, b_vec, **dists)
    else:
        AddSmearedEventsToMap(m, l_vec, b_vec, **dists)
    return m



def ConvolveUsingAlm(map_in, psf_alm):
    """Convolve a map using a set of pre-computed ALM

    Parameters
    ----------
    map_in : array_like
        HEALPix map to be convolved
    psf_alm : array_like
        The ALM represenation of the PSF

    Returns
    -------
    map_out : array_like
        The smeared map
    """
    norm = map_in.sum()
    nside = hp.pixelfunc.npix2nside(map_in.size)
    almmap = hp.sphtfunc.map2alm(map_in)
    almmap *= psf_alm
    outmap = hp.sphtfunc.alm2map(almmap, nside)
    outmap *= norm / outmap.sum()
    return outmap
