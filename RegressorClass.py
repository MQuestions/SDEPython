# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 21:48:06 2020

@author: Akhi
"""
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from OrnsteinUhlenbeckRegressor import OrnsteinUhlenbeckRegressor

class Regressor(object):
    OU = OrnsteinUhlenbeckRegressor
    
    @staticmethod
    def YearFraction(Data):
        days = np.asarray(range(1,len(Data)+1))/len(Data)
        hours = np.asarray(range(1, 25))/24

        yearFract = np.zeros((len(Data), 24))
        for i in range(len(Data)):
            for j in range(24):
                yearFract[i,j] = days[i]*hours[j]

        yF = np.transpose(yearFract)

        return yF
    
    @staticmethod
    def SeasonalModel(TimeSteps):
        return np.c_[np.sin(2*np.pi*TimeSteps), np.cos(2*np.pi*TimeSteps), np.sin(4*np.pi*TimeSteps), np.cos(4*np.pi*TimeSteps), np.sin(6*np.pi*TimeSteps), np.cos(6*np.pi*TimeSteps), TimeSteps, np.ones(len(TimeSteps))]

    @staticmethod
    def SeasonalCalibration(Prices, Time):
#        tmp = self.SeasonalModel(Time)
#        print(tmp.shape)
#        print(Prices.shape)
        return np.linalg.pinv(Regressor.SeasonalModel(Time)).dot(Prices)

    @staticmethod
    def SeasonalityAdjustment(Prices_i, Time_i):
        Y = Regressor.SeasonalModel(Time_i)
        SP = Regressor.SeasonalCalibration(Prices_i, Time_i)
        return np.dot(Y, SP)

    @staticmethod
    def SeasonalityAdjustments(prices):
        time = Regressor.YearFraction(prices)
        SM = np.zeros((361, 24))
        for i in range(24):
            SM[:,i] = Regressor.SeasonalityAdjustment(prices[:,i], time[i])
        return SM


        
        
        
        
        
