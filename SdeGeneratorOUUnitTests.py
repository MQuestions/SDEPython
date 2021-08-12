# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 22:10:29 2020

@author: Akhi
"""

import unittest
from SdeGenerator import GridParameters, SdeGenerator
from RegressorClass import Regressor
import matplotlib.pyplot as plt

import numpy as np

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

#    OU(x0, mu, sigma, theta):
    def test_OUWithZeroSigma(self):
        process1 = self.SdeGenerator1.OU(0, 0, 0, 1)
        process2 = self.SdeGenerator1.OU(0, 1, 0, 1)
        process3 = self.SdeGenerator2.OU(1, 1, 0, 1)

        self.assertAlmostEqual(process1[-1], 0)
        self.assertAlmostEqual(process2[-1], 1 - np.exp(-2))
        self.assertAlmostEqual(process3[-1], 1)
        
    # Running many simulations, it makes sense to collect them all into a matrix.
    def test_OUSimMatrixHasCorrectSize(self):
        OUSimMatrix1 = self.SdeGenerator1.OUMany(0, 0, 1, 1, 10)
        self.assertEqual(OUSimMatrix1.shape, (10, 200))
        
    def test_SdeWithZeroMu_VerifyMean(self):
        #OUMany(self, x0, mu, sigma, theta, Sims):
        Sims1 = 500
        Simulations1 = self.SdeGenerator1.OUMany(0, 0, 1, 1, Sims1)
        AvgSimulations1 = SdeGenerator.AverageOverSims(Simulations1)
        self.assertGreaterEqual(self.z_value, AvgSimulations1[0])
        self.assertGreaterEqual(self.z_value, AvgSimulations1[50])            
        self.assertGreaterEqual(self.z_value, AvgSimulations1[150])            
        self.assertGreaterEqual(self.z_value, AvgSimulations1[-1])
        
        Sims2 = 500
        Simulations2 = self.SdeGenerator2.OUMany(1, 0, 1, 1, Sims2)
        AvgSimulations2 = SdeGenerator.AverageOverSims(Simulations2)
        difference = AvgSimulations2[-1] - np.exp(-5)
        self.assertGreaterEqual(self.z_value, difference)
        
        Sims3 = 500
        Simulations3 = self.SdeGenerator3.OUMany(1, 0, 1, 0, Sims3)
        AvgSimulations3 = SdeGenerator.AverageOverSims(Simulations3)
        expectedMean= np.exp(-5)
        diff = expectedMean - AvgSimulations3[-1]        
        self.assertGreaterEqual(self.z_value, diff)
        
        Sims1 = 500
        Simulations1 = self.SdeGenerator1.OUMany(1, 2, 3, 4, Sims1)
        AvgSimulations1 = SdeGenerator.AverageOverSims(Simulations1)
        expectedMean = np.exp(-2)*1 + 2*(1 - np.exp(-4*2))
        diff = expectedMean - AvgSimulations1[-1]
        self.assertGreaterEqual(self.z_value, diff)
        
    def test_SdeWithZeroTheta_VerifyMean(self):
        #OUMany(self, x0, mu, sigma, theta, Sims):
        Sims1 = 500
        Simulations1 = self.SdeGenerator1.OUMany(0, 1, 1, 0, Sims1)
        AvgSimulations1 = SdeGenerator.AverageOverSims(Simulations1)
        diff = AvgSimulations1[-1]-1
        self.assertGreaterEqual(self.z_value, diff)
        
        Sims2 = 500
        Simulations2 = self.SdeGenerator2.OUMany(1, 1, 1, 0, Sims2)
        AvgSimulations2 = SdeGenerator.AverageOverSims(Simulations2)
        difference = AvgSimulations2[-1] - 1
        self.assertGreaterEqual(self.z_value, difference)
        
        Sims3 = 500
        Simulations3 = self.SdeGenerator3.OUMany(1, 2, 3, 0, Sims3)
        AvgSimulations3 = SdeGenerator.AverageOverSims(Simulations3)
        diff = AvgSimulations3[-1] - 1     
        self.assertGreaterEqual(self.z_value, diff)
        
    def test_SdeWithNonZeroMuSig_VerifyMean(self):
            
        Sims1 = 500
        Simulations1 = self.SdeGenerator1.OUMany(0, 1, 1, 1, Sims1)
        AvgSimulations1 = SdeGenerator.AverageOverSims(Simulations1)
        difference = AvgSimulations1[-1] - (1-np.exp(-2))
        self.assertGreaterEqual(self.z_value, difference)
        
        Sims2 = 500
        Simulations2 = self.SdeGenerator2.OUMany(1, 1, 1, 1, Sims2)
        AvgSimulations2 = SdeGenerator.AverageOverSims(Simulations2)
        difference1 = AvgSimulations2[-1] - (np.exp(-5) + (1 - np.exp(-5)))
        self.assertGreaterEqual(self.z_value, difference1)
        
        Sims3 = 800
        Simulations3 = self.SdeGenerator3.OUMany(1, 1, 1, 0, Sims3)
        AvgSimulations3 = SdeGenerator.AverageOverSims(Simulations3)
        self.assertGreaterEqual(self.z_value, AvgSimulations3[-1])
        
        #OUMany(x0, mu, sigma, theta, Sims):
        Simulations1 = self.SdeGenerator1.OUMany(10, 2, 4, 8, 500)
        expectedVariance = 4 * 4 * (10 - np.exp(-2 * 8 * self.SdeGenerator1.t_end)) / (2 * 8) 
        difference = SdeGenerator.VarSims(Simulations1)[-1] - expectedVariance
        self.assertGreaterEqual(self.z_value, difference)

    def test_SdeWithZeroMu_VerifyVariance(self):
        #OUMany(self, x0, mu, sigma, theta, Sims):
        Sims1 = 500
        Simulations1 = self.SdeGenerator1.OUMany(0, 0, 1, 1, Sims1)
        expectedVariance = (1 - np.exp(-2*2))/(2)
        difference = SdeGenerator.VarSims(Simulations1)[-1] - expectedVariance
        self.assertGreaterEqual(self.z_value, difference)
        
        Sims2 = 500
        Simulations2 = self.SdeGenerator2.OUMany(1, 0, 1, 1, Sims2)
        expectedVariance1 = (1 - np.exp(-2*5))/(2)
        difference1 = SdeGenerator.VarSims(Simulations2)[-1] - expectedVariance1
        self.assertGreaterEqual(self.z_value, difference1)
        
        Sims3 = 800
        Simulations3 = self.SdeGenerator3.OUMany(1, 0, 1, 0, Sims3)
        expectedVariance2 = 10
        difference2 = SdeGenerator.VarSims(Simulations3)[-1] - expectedVariance2
        self.assertGreaterEqual(self.z_value, difference2)
        
    def test_SdeWithZeroTheta_VerifyVariance(self):
        #OUMany(self, x0, mu, sigma, theta, Sims):
        Sims1 = 500
        Simulations1 = self.SdeGenerator1.OUMany(0, 0, 1, 0, Sims1)
        expectedVariance = 2
        difference = SdeGenerator.VarSims(Simulations1)[-1] - expectedVariance
        self.assertGreaterEqual(self.z_value, difference)
        
        Sims2 = 500
        Simulations2 = self.SdeGenerator2.OUMany(1, 0, 1, 0, Sims2)
        expectedVariance1 = 5
        difference1 = SdeGenerator.VarSims(Simulations2)[-1] - expectedVariance1
        self.assertGreaterEqual(self.z_value, difference1)
   
    def test_SdeWithNonZeroMuSig_VerifyVariance(self):
        
        Sims1 = 500
        Simulations1 = self.SdeGenerator1.OUMany(0, 1, 1, 1, Sims1)
        expectedVariance = (1 - np.exp(-2*2))/(2)
        difference = SdeGenerator.VarSims(Simulations1)[-1] - expectedVariance
        self.assertGreaterEqual(self.z_value, difference)
        
        Sims2 = 500
        Simulations2 = self.SdeGenerator2.OUMany(1, 1, 1, 1, Sims2)
        expectedVariance1 = (1 - np.exp(-2*5))/(2)
        difference1 = SdeGenerator.VarSims(Simulations2)[-1] - expectedVariance1
        self.assertGreaterEqual(self.z_value, difference1)
        
        Sims3 = 800
        Simulations3 = self.SdeGenerator3.OUMany(1, 1, 1, 0, Sims3)
        expectedVariance2 = 10
        difference2 = SdeGenerator.VarSims(Simulations3)[-1] - expectedVariance2
        self.assertGreaterEqual(self.z_value, difference2)
        
#    OU(self, x0, mu, sigma, theta):    
#    def test_RegressingOU(self):
        #process1 = self.SdeGenerator1.OU(5, 1, 0, 3)
#        Simulations1 = self.SdeGenerator1.OUMany(0, 1, 0.01, 10, 500)
#        avg = SdeGenerator.AverageOverSims(Simulations1)
#        coeff = Regressor.OU(avg, self.SdeGenerator1.dt)
        #print(self.SdeGenerator1.dt)
        #print("\ntheta = " + str(coeff[0]))
        #print("mu = " + str(coeff[1]))
        #print("sd = " + str(coeff[2]))
        #print("sd / sqrt(dt) = " + str(coeff[2] / np.sqrt(self.SdeGenerator1.dt)))

        #    OUMany(self, x0, mu, sigma, theta, Sims):
    def test_RegressingOUModified(self):
        gParameters = GridParameters(0, 5, 500) # dt = 0.025
        sdeGenerator = SdeGenerator(gParameters)
        sim = 150
        
        thetaInput =  16
        muInput = 0.19
        sigmaInput = 1.10
        
        muTotal = 0.0
        thetaTotal = 0.0
        sigmaTotal = 0.0
        for i in range(sim):
            samplePath = sdeGenerator.OU(3.0, muInput, sigmaInput, thetaInput)
            coeff = Regressor.OU.Mle(samplePath, sdeGenerator.dt)
            
            thetaTotal += coeff[0]
            muTotal += coeff[1]
            sigmaTotal += coeff[2]

        avg_theta = thetaTotal / sim
        avg_mu = muTotal / sim
        avg_sigma = sigmaTotal / sim
        
        #print("avg theta = " + str(avg_theta))
        #print("avg mu = " + str(avg_mu))
        #print("avg sigma = " + str(avg_sigma))
        
    def test_RegressingOUJackKnife(self):
        gParameters = GridParameters(0, 1, 500) # dt = 0.025
        sdeGenerator = SdeGenerator(gParameters)
        
        sim = 150
        
        thetaInput =  20
        muInput = 1.2
        sigmaInput = 0.5
        
        muTotal = 0.0
        thetaTotal = 0.0
        sigmaTotal = 0.0
        for i in range(sim):
            samplePath = sdeGenerator.OU(3.0, muInput, sigmaInput, thetaInput)
            coeff = Regressor.OU.JackKnife(samplePath, 15, sdeGenerator.dt)
            
            thetaTotal += coeff[0]
            muTotal += coeff[1]
            sigmaTotal += coeff[2]

        avg_theta = thetaTotal / sim
        avg_mu = muTotal / sim
        avg_sigma = sigmaTotal / sim
        
        print("avg theta = " + str(avg_theta))
        print("avg mu = " + str(avg_mu))
        print("avg sigma = " + str(avg_sigma))
            
#        Simulations1 = self.SdeGenerator1.OUMany(0, 5, 2, 1, 500)
  #      avg = SdeGenerator.AverageOverSims(Simulations1)
        
#        plt.plot(sdeGenerator.t_domain, avg)
        
#        print("\n diss one\n theta = " + str(coeff[0]))
#        print("mu = " + str(coeff[1]))
#        print("sd = " + str((coeff[2])) + "--\n")
        #print("sd / sqrt(dt) = " + str(coeff[2] / np.sqrt(self.SdeGenerator1.dt)))
        
#    def test_RegressingOUModifiedOLSExplicit(self):
#        process1 = self.SdeGenerator1.OU(5, 8, 0.025, 1)
#        Simulations1 = self.SdeGenerator1.OUMany(0, 5, 2, 1, 500)
  #      avg = SdeGenerator.AverageOverSims(Simulations1)
#        coeff = Regressor.OUModifiedExplicit(process1, self.SdeGenerator1.dt)
#        print(self.SdeGenerator1.dt)
#        print("\ntheta = " + str(coeff[0]))
#        print("mu = " + str(coeff[1]))
#        print("sd = " + str((coeff[2])))
        #print("sd / sqrt(dt) = " + str(coeff[2] / np.sqrt(self.SdeGenerator1.dt)))
        
        
        
if __name__ == '__main__':
    unittest.main()
    
    
    
    