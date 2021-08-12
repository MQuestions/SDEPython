# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 12:41:05 2020

@author: Akhi
"""

import unittest
from StochasticProcess import StochasticProcess

import numpy as np

class TestingSdeGen(unittest.TestCase):
#    Process1 = [1,2,5,8]
    
    def test_average(self):
        # First create some (fake) data...
        Process1 = [1,2,5,8]
        Process2 = [1,2,-5,8]
        Process3 = [1,2,0,8]

        # Then "turn/wrap" it into a stochastic process object
        # StochasticProcess is the class, we "construct" it via inputting the data in the bracks
        SP1 = StochasticProcess(Process1)
        SP2 = StochasticProcess(Process2)
        SP3 = StochasticProcess(Process3)
        
        # Assert average holds
        self.assertAlmostEqual(SP1.avg(), 4)
        self.assertAlmostEqual(SP2.avg(), 1.5)
        self.assertAlmostEqual(SP3.avg(), 2.75)
        
    def test_maximum(self):
        # First create some (fake) data...
        Process1 = [1,2,5,8]

        # Then "turn/wrap" it into a stochastic process object
        # StochasticProcess is the class, we "construct" it via inputting the data in the bracks
        SP1 = StochasticProcess(Process1)
        
        # Assert average holds
        self.assertAlmostEqual(SP1.maximum(), 8)

    def test_minimum(self):
        # First create some (fake) data...
        Process1 = [1,2,5,8]
        Process2 = [1,2,-5,8]
        Process3 = [1,2,0,8]

        # Then "turn/wrap" it into a stochastic process object
        # StochasticProcess is the class, we "construct" it via inputting the data in the bracks
        SP1 = StochasticProcess(Process1)
        SP2 = StochasticProcess(Process2)
        SP3 = StochasticProcess(Process3)
        
        # Assert average holds
        self.assertAlmostEqual(SP1.minimum(), 1)
        self.assertAlmostEqual(SP2.minimum(), -5)
        self.assertAlmostEqual(SP3.minimum(), 0)

    def test_CheckNegative(self):
        # First create some (fake) data...
        Process1 = [1,2,5,8]
        Process2 = [1,2,-5,8]
        Process3 = [1,2,0,8]

        # Then "turn/wrap" it into a stochastic process object
        # StochasticProcess is the class, we "construct" it via inputting the data in the bracks
        SP1 = StochasticProcess(Process1)
        SP2 = StochasticProcess(Process2)
        SP3 = StochasticProcess(Process3)
        
        self.assertAlmostEqual(SP1.checkNegative(), False)
        self.assertAlmostEqual(SP2.checkNegative(), False)
        self.assertAlmostEqual(SP3.checkNegative(), False)


        
    
if __name__ == '__main__':
    unittest.main()