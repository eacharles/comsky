"""
Unit tests for PDF class
"""
import sys
import os
import numpy as np
import scipy.stats as sps
import healpy as hp
import unittest
import comsky

NSIDE = 256

class UtilsTestCase(unittest.TestCase):

    def setUp(self):
        """
        Make any objects that are used in multiple tests.
        """
        self.files = []

    def tearDown(self):
        "Clean up any mock data files created by the tests."
        for ff in self.files:
            os.unlink(ff)

    def test_point_source(self):
        nSrc = 1000
        l = 184.55746
        b = -5.78436
        mSrc = comsky.utils.MakePointSource(nSrc, NSIDE, l, b)
        sampSrc = comsky.utils.MakePointSource(nSrc, NSIDE, l, b, sample=True)
        
    def test_galactic(self):
        nIso = 1000
        mIso = comsky.utils.MakeIsotropicBackground(nIso, NSIDE)
        sampIso = comsky.utils.MakeIsotropicBackground(nIso, NSIDE, sample=True)
        
    def test_isotropic(self):
        nGal = 10000
        mGal = comsky.utils.MakeGalacticBackground(nGal, NSIDE)
        sampGal = comsky.utils.MakeGalacticBackground(nGal, NSIDE, sample=True)
        
    def testPSFConvolve(self):
        nSrc = 10000
        l = 0
        b = 90
        mPSF = comsky.utils.MakePointSource(nSrc, NSIDE, l, b)
        almPSF = hp.sphtfunc.map2alm(mPSF)
        vecNorth = np.array([0., 0., 1.])
        mGalTrue = np.zeros(hp.pixelfunc.nside2npix(NSIDE))
        sel = hp.query_disc(NSIDE, vecNorth, np.radians(92))
        mGalTrue[sel] += 1.
        sel = hp.query_disc(NSIDE, vecNorth, np.radians(88))
        mGalTrue[sel] -= 1.
        mGalConv = comsky.utils.ConvolveUsingAlm(mGalTrue, almPSF)

    def testSamplePoints(self):
        npts = 100
        l = 0
        b = 0
        rad = 30
        phi_dist = sps.uniform(scale=2*np.pi)
        theta, phi = hp.pixelfunc.lonlat2thetaphi(l, b)
        pts = comsky.utils.SamplePointsFromRings(phi, theta, np.radians(rad), phi_dist)


    def testAddRing(self):
        nside = 256
        m = np.zeros((hp.pixelfunc.nside2npix(nside)))
        l = 0
        b = 0
        theta, phi = hp.pixelfunc.lonlat2thetaphi(l, b)
        ipix = hp.pixelfunc.ang2pix(nside, theta, phi)
        comsky.utils.AddRingToMap(m, ipix, nside, radius=30, width=5)

        
if __name__ == '__main__':
    unittest.main()
