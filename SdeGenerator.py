# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 21:54:59 2020

@author: Akhi
"""
import numpy as np
from StochasticProcess import *

class GridParameters(object):
    # t0, t_end obv
    # N - number of total steps (including initial condition x0)
    def __init__(self, t0, t_end, N):
        self.t0 = t0
        self.t_end = t_end
        self.N = N
        

class SdeGenerator(object):
    # x0 - initial condition
    # mu - constant drift
    # sigma - volatility
    # theta - mean reversion
    
    def __init__(self, gridParameters):
        self.t0 = gridParameters.t0
        self.t_end = gridParameters.t_end
        self.N = gridParameters.N
        self.t_domain = np.linspace(self.t0, self.t_end, self.N)
        # dt is for the next (N-1) steps hence together with x0 makes N steps.
        self.dt = float(self.t_end - self.t0) / (self.N - 1)
        self.sqrt_dt = np.sqrt(self.dt)

    # This is SDE for dX = mu dt + sigma dW, where mu and sigma are constants    
    def ConstantParameters(self, x0, mu, sigma):        
        dW = np.random.normal(0, self.sqrt_dt, self.N)
        W = np.cumsum(dW)
        W[0] = 0
        
        process = np.zeros(self.N)
        process = x0 + np.linspace(0, self.N - 1, self.N)*mu*self.dt + sigma*W
        return process
    
    def ConstantParametersMany(self, x0, mu, sigma, Sims):
        MyStupidMatrix = np.zeros((Sims, self.N))
        for sim in range(Sims):
            MyStupidMatrix[sim] = self.ConstantParameters(x0, mu, sigma)
    
        return MyStupidMatrix       
        
    def avg(self, x0, mu, sigma):
        Process = self.ConstantParameters(x0, mu, sigma)
        return np.average(Process)
    
    def std(self, x0, mu, sigma):
        Process = self.ConstantParameters(x0, mu, sigma)
        return np.std(Process)
    
    def maximum(self, x0, mu, sigma):
        Process = self.ConstantParameters(x0, mu, sigma)
        return np.max(Process)
    
    def minimum(self, x0, sigma, mu):
        Process = self.ConstantParameters(x0, mu, sigma)
        return np.min(Process)
    
#    def checkNegative(self, x0, mu, sigma):
#        Process = self.ConstantParameters(x0, mu, sigma)
#        return np.min(Process) < 0
    
#    def Correlation(self, x0, mu, sigma):
#        Process0 = self.ConstantParameters(x0, mu, sigma)
#        Process1 = self.ConstantParameters(x0, mu, sigma)
#        return np.corrcoef(Process0, Process1)
    
    # Inputting a matrix of simulationsm, np.sum(, 0) is average over sim!
    @staticmethod
    def AverageOverSims(simulations):
        return np.average(simulations, 0)
        
    # Remember 1: static method applies to the "class" i.e. to all particular cases hence "better"
    # Remember 2: you wanted the danged matrix, the "(sims,0)" axis yerza.
    @staticmethod
    def VarSims(simulations):
        return np.var(simulations, 0)
    
    # A (standard) Brownian motion is one that starts at the origin, no drift and sigma 1
#    @staticmethod
    def BrownianMotion(self):
        return StochasticProcess(self.ConstantParameters(0, 0, 1))
        
    # A (standard) Brownian motion with initial x0
#    @staticmethod
    def BrownianMotionWithInitial(self, x0):
        return StochasticProcess(self.ConstantParameters(x0, 0, 1))
        
    def GBM(self, x0, mu, sigma):
        process = np.zeros(self.N)
        process[0] = x0
        
        dW = self.sqrt_dt * np.random.normal(0, 1, self.N)
        drift = (mu - sigma**2 * 0.5)
        
        t = list(range(self.t0, self.t_end+1))        
        for i in range(1, self.t_domain.size):
            process[i] = process[i - 1] * np.exp(drift * t[i] + sigma * dW[i])
            
        return process

    def GBMany(self, x0, mu, sigma, Sims):
        GBMSimMatrix = np.zeros((Sims, self.N))
        for sim in range(Sims):
            GBMSimMatrix[sim] = self.GBM(x0, mu, sigma, theta)
    
        return GBMSimMatrix
        
    #   dX = theta*(mu - X)*dt + sigma*dW
    def OU(self, x0, mu, sigma, theta):
        # if theta super small, it is essentially a BM.
        if (abs(theta) < 1e-5): 
            return self.ConstantParameters(x0, 0, sigma)
        
        if (abs(sigma) < 1e-5): 
            return self.OU_DeterminsiticPart(x0, mu, sigma, theta)
 
        process = np.zeros(self.N)
        process[0] = x0
        exp_minus_lambda_deltat = np.exp(-theta*self.dt);

        dW =  np.sqrt((1-np.exp(-2*theta* self.dt))/(2*theta)) * np.random.normal(0, 1, self.N)

        for i in range(1, self.t_domain.size):
            process[i] = process[i - 1]*exp_minus_lambda_deltat + mu*(1-exp_minus_lambda_deltat) + sigma*dW[i]
        
        return process
        
    def OU_DeterminsiticPart(self, x0, mu, sigma, theta):
            return np.exp(-theta*self.t_domain)*x0 + mu*(1 - np.exp(-theta*self.t_domain))
        
    def OUMany(self, x0, mu, sigma, theta, Sims):
        OUSimMatrix = np.zeros((Sims, self.N))
        for sim in range(Sims):
            OUSimMatrix[sim] = self.OU(x0, mu, sigma, theta)
    
        return OUSimMatrix 
        
    #def ExponentialArrivalTime(self, lambda, tau):
   #     process = np.zeros(self.N)
        
      

        
        
        
        
        
        
        
        
        