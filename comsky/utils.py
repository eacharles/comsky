"""Utility functions"""

import numpy as np
import healpy as hp
import scipy.stats as sps

def AddRingToMap(m, pix, nside, radius=30, width=5): 
    """Given a healpix map and a sky direction (given here as the healpix pixel index),
    add the back-projected compton cone around that pixel to the map. 
    That is done by finding the pixels in a ring (with given radius and width) around the given direction, 
    and adding 1 to the map value of the pixels in that ring. (Might want to normalize it by the ring area eventually.)
    
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



def GetNptsForResolution(nside, alpha=1, factor=2):
    return np.ceil(factor*2*np.pi*alpha / hp.pixelfunc.nside2resol(nside)).astype(int)


def GetLociPoints(lon, colat, alpha, phi):

    # https://math.stackexchange.com/questions/643130/circle-on-sphere
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

    phi_vec = np.linspace(0, 2*np.pi, npts)
    return GetLociPoints(lon, colat, alpha, phi_vec)


def SamplePointsFromRing(lon, colat, alpha, phi_dist):
    npts = lon.size
    phi = phi_dist.rvs(npts)
    return GetLociPoints(lon, colat, alpha, phi)
    


def AddLociToMap(m, lon, colat, alpha, factor=2, normed=False):

    nside = hp.pixelfunc.npix2nside(m.size)
    npts = GetNptsForResolution(nside, alpha=alpha, factor=factor)

    vecs = GetLociRing(lon, colat, alpha, npts)    
    ipix = hp.pixelfunc.vec2pix(nside, vecs[0], vecs[1], vecs[2])
    
    weight = 1.
    if normed:
        weight /= float(npts)

    for ipix_ in ipix:
        m[ipix_] += weight


def SafeRVS(dist, nevt):
    if np.isscalar(dist):
        return np.ones(nevt)*dist
    return dist.rvs(size=nevt)
    

def AddSmearedEventsToMap(m, l, b, alpha_true_dist, delta_alpha_dist, phi_true_dist):

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


def AddSampledEventsToMap(m, l, b, alpha_true_dist, delta_alpha_dist, phi_true_dist, phi_sample_dist):    
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
    print(ipix.size)
    for ipix_ in ipix:
        m[ipix_] += 1


def GetDefaultDistributions(radius=30., width=5.):

    sigma = width/2.3548
    return dict(alpha_true_dist=sps.norm(loc=np.radians(radius), scale=np.radians(width)),
                    delta_alpha_dist=sps.norm(scale=np.radians(sigma)),
                    phi_true_dist=sps.uniform(scale=2*np.pi))

    
    
def MakePointSource(nPS, nside, l, b, radius=30., width=5., sample=False):

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
        
        
def MakeIsotropicBackground(nIso, nside, radius=30, width=5, sample=False):

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


def MakeGalacticBackground(nGal, nside, radius=30, width=5, sample=False):
    
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



def ConvolveUsingAlm(map1, psf_alm):
    norm = map1.sum()
    nside = hp.pixelfunc.npix2nside(map1.size)
    almmap = hp.sphtfunc.map2alm(map1)
    almmap *= psf_alm
    outmap = hp.sphtfunc.alm2map(almmap, nside)
    outmap *= norm / outmap.sum()
    return outmap

