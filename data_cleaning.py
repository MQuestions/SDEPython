# -*- coding: utf-8 -*-
"""
Created on Sun Mar 01 12:53:57 2020

@author: Akhi
"""

import numpy as np
import pandas as pd
import math
import scipy as sp
from scipy.linalg import solve
import matplotlib.pyplot as plt
from pandas import ExcelWriter
from pandas import ExcelFile

Prices = pd.read_csv("NordSysy2019Hourly.csv")

#filtered_prices = Price[Prices > ]

def get_diffs(P):
    for i in range(0, len(P)):
        pcolumn = P[i]
        pdiffs = np.diff(pcolumn)
        pdiffs = np.insert(pdiffs, [0], 0)
        P[i] = pdiffs
        
    return P

def cleanse_outliers(P):
    for i in range(0, len(P)):
        mean = np.mean(P[i])
        standard_deviation = np.std(P[i])
        for j in range(0, len(P[i])):
            if abs(P[i][j]) > standard_deviation:
                print("Changing {0} to {1}".format(P[i][j], mean))
                P[i][j] = mean
                print("New value: {0}".format(P[i][j]))
                
    return P

def zero_division_two_dimensional(arr1,arr2):
    results = np.zeros((len(arr1), len(arr1.transpose())))
    for i in range(0, len(arr1.transpose())):
        for j in range(0, len(arr1)):
            if arr2[j][i] == 0:
                results[j][i] = 0
            else:
                results[j,i] = arr1[j][i] / arr2[j][i]
                
    return results

P = np.array(Prices)[0:361,1:25].transpose()
PDiffs = get_diffs(P)
cleansed_pdiffs = cleanse_outliers(PDiffs)