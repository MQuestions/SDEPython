# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 22:10:29 2020

@author: Akhi
"""

import unittest
from SdeGenerator import GridParameters, SdeGenerator

import numpy as np

def RelError(real, fake):
    return abs(real - fake) / real

class TestingSdeGen(unittest.TestCase):
    
    # This sets up the SDE generators with some fixed grid parameters
    def setUp(self):
        # Tolerance value for some testing. z-value of 2 means ~97% of Normal dist
        self.tolerance = 1e-5
        self.z_value = 2 

        #GridParameters(t0, t_end, N)        
        gParameters0 = GridParameters(0, 1, 2)
        gParameters1 = GridParameters(0, 2, 200)
        gParameters2 = GridParameters(0, 5, 300)
        gParameters3 = GridParameters(0, 10, 400)
        self.OneStepperSde = SdeGenerator(gParameters0)
        self.SdeGenerator1 = SdeGenerator(gParameters1)
        self.SdeGenerator2 = SdeGenerator(gParameters2)
        self.SdeGenerator3 = SdeGenerator(gParameters3)

#    ConstantParameters(self, x0, mu, sigma):
    def test_SdeLengthIsOK(self):
        process1 = self.SdeGenerator1.ConstantParameters(1, 0, 0)
        process2 = self.SdeGenerator2.ConstantParameters(1, 1, 0)
        process3 = self.SdeGenerator3.ConstantParameters(1, 0, 1)
        
        self.assertAlmostEqual(process1.size, 200)
        self.assertAlmostEqual(process2.size, 300)
        self.assertAlmostEqual(process3.size, 400)
        
#    def test_CheckNegative(self):
#        processCheck = self.SdeGenerator1.checkNegative(0, 1, 1)
#        
#        self.assertAlmostEqual((processCheck), False)
    
    def test_InitialValueIsCorrect(self):
        process1 = self.SdeGenerator1.ConstantParameters(1, 0, 0)
        process2 = self.SdeGenerator2.ConstantParameters(2, 1, 0)
        process3 = self.SdeGenerator3.ConstantParameters(3, 0, 1)
        
        self.assertAlmostEqual(process1[0], 1)
        self.assertAlmostEqual(process2[0], 2)
        self.assertAlmostEqual(process3[0], 3, 2)
        
    # Sigma = 0 means testing dX = mu dt => X_t = x0 + mu(t_end-t0)
    def test_SdeWithZeroSigma(self):
        process1 = self.SdeGenerator1.ConstantParameters(0, 0, 0)
        process2 = self.SdeGenerator2.ConstantParameters(1, 1, 0)
        process3 = self.SdeGenerator3.ConstantParameters(2, 2, 0)

        self.assertAlmostEqual(process1[-1], 0)
        self.assertAlmostEqual(process2[-1], 6)
        self.assertAlmostEqual(process3[-1], 22)

    # Running many simulations, it makes sense to collect them all into a matrix.
    def test_MyStupidMatrixHasCorrectSize(self):
        myStupidMatrix1 = self.SdeGenerator1.ConstantParametersMany(0, 0, 1, 10)
        self.assertEqual(myStupidMatrix1.shape, (10, 200))
        
        myStupidMatrix2 = self.SdeGenerator2.ConstantParametersMany(0, 0, 1, 15)
        self.assertEqual(myStupidMatrix2.shape, (15, 300))

        myStupidMatrix3 = self.SdeGenerator3.ConstantParametersMany(0, 0, 1, 0)
        self.assertEqual(myStupidMatrix3.shape, (0, 400))
        
    # ConstantParametersMany(x0, mu, sigma, Sims)
    # Testing mu = 0, sigma != 0 
    # i.e. testing Brownian motion (BM) with solution Xt = x0 + sigma*W(t)
    # Pathwise is "difficult" because it is random hence we test its statistical properties

    # BM test 1: Test mean
    # While the generated SDE has N points, we do NOT check all the points have average zero within tolerance.
    # This is because even if ANY GIVEN point has say 95% of passing, 
    # the probability of passing ALL is not 95% and is much lower (probs geometrically) 
    # Hence test at a few selected points
    def test_SdeWithZeroMu_VerifyMean(self):
        Sims = 500
        Simulations = self.OneStepperSde.ConstantParametersMany(0, 0, 1, Sims)
        AvgSimulations = SdeGenerator.AverageOverSims(Simulations)
        for sim in AvgSimulations:
            self.assertTrue(self.z_value >= sim)
            
        Sims1 = 500
        Simulations1 = self.SdeGenerator1.ConstantParametersMany(0, 0, 1, Sims1)
        AvgSimulations1 = SdeGenerator.AverageOverSims(Simulations1)
        self.assertGreaterEqual(self.z_value, AvgSimulations1[0])
        self.assertGreaterEqual(self.z_value, AvgSimulations1[50])            
        self.assertGreaterEqual(self.z_value, AvgSimulations1[150])            
        self.assertGreaterEqual(self.z_value, AvgSimulations1[-1])
        
        Sims2 = 750
        Simulations2 = self.SdeGenerator1.ConstantParametersMany(0, 0, 1, Sims2)
        AvgSimulations2 = SdeGenerator.AverageOverSims(Simulations2)
        self.assertGreaterEqual(self.z_value, AvgSimulations2[0])
        self.assertGreaterEqual(self.z_value, AvgSimulations2[50])            
        self.assertGreaterEqual(self.z_value, AvgSimulations2[150])            
        self.assertGreaterEqual(self.z_value, AvgSimulations2[-1])
        
    def test_SdeWithNonzeroMuNonzeroSig(self):
        Sims = 500
        Simulations1 = self.SdeGenerator1.ConstantParametersMany(0, 1, 1, Sims)
        AvgSimulations1 = SdeGenerator.AverageOverSims(Simulations1)
        difference = AvgSimulations1[-1] - 2
        self.assertGreaterEqual(self.z_value, difference)

        Simulations2 = self.SdeGenerator2.ConstantParametersMany(0, 3, 5, Sims)
        AvgSimulations2 = SdeGenerator.AverageOverSims(Simulations2)
        difference2 = AvgSimulations2[-1] -15
        self.assertGreaterEqual(self.z_value, difference2)

    #ConstantParametersMany(x0, mu, sigma, Sims)
    def test_SdeVarianceWithConstParameters(self):
        Sims = 500
        Simulations1 = self.SdeGenerator1.ConstantParametersMany(0, 1, 1, Sims)
        expectedVariance = 2 # now we need std^2 (Jumydia^2)ehehehhe ><"too good
         #junmydia
         # Remember 3: this is at every time step... "=h yeah -1 = last
        difference = SdeGenerator.VarSims(Simulations1)[-1] - expectedVariance #i.e. simulation var minus theory var.
        self.assertGreaterEqual(self.z_value, difference) # probability test 'ere.

        # Ak Ak do next one - kinda C&P...
        Simulations2 = self.SdeGenerator2.ConstantParametersMany(0, 3, 5, Sims)
        expectedVariance2 = 125
        difference2 = SdeGenerator.VarSims(Simulations2)[-1] - expectedVariance2
        self.assertGreaterEqual(self.z_value, difference2)
        
        Simulations3 = self.SdeGenerator3.ConstantParametersMany(0, 4, 4, Sims)
        expectedVariance3 = (self.SdeGenerator3.t_end - self.SdeGenerator3.t0) * 16 #heehee
        difference3 = SdeGenerator.VarSims(Simulations3)[-1] - expectedVariance3
        self.assertGreaterEqual(self.z_value, difference3)

if __name__ == '__main__':
    unittest.main()
    
    
    
    