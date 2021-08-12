# -*- coding: utf-8 -*-
"""
Created on Sun Mar 08 15:31:31 2020

@author: Akhi
"""
import numpy as np

# we implicitly assumes for ALL (one dim) stochastic process
class StochasticProcess:
    
    def __init__(self, Process):
        self.Process = Process
        
    def avg(self):
        return np.average(self.Process)
    
    def std(self):
        return np.std(self.Process)
    
    def maximum(self):
        return np.max(self.Process)
    
    def minimum(self):
        return np.min(self.Process)
    
    def NumberOfJumps(self):
        return 0
        
    def checkNegative(self):
        return np.min(self.Process) < 0
    
    def __repr(self):
        return "generic process (tmp)"
        
    def __add__(self, otherStochasticProcess):
        return StochasticProcess(self.Process - otherStochasticProcess.Process)
        
    def __sub__(self, otherStochasticProcess):
        return StochasticProcess(self.Process - otherStochasticProcess.Process)

    @staticmethod    
    def Correlation(stochasticProcessObject1, stochasticProcessObject2):
        return np.corrcoef(stochasticProcessObject1.Process, stochasticProcessObject2.Process)

    @staticmethod
    def WrapIntoSP(process):
        return StochasticProcess(process)
        
        #Tesing these - alright but doing something epiks