# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 09:50:55 2020

@author: shimk
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.metrics import mean_squared_error


# load the fiited DNN model for regression of SOP
new_model = keras.models.load_model('Trained_DNN_FarUserOTS.h5') # call the fitted ML model for regression

# network parameters
MM = 4
PS_dB = [-20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
PN_dB = 10
PF_dB = 10
PE_dB = 10
#

dSN = 0.2
dSF = 1
dSE = 1
#
dNE = 0.5
dFE = 0.5
#
thetaN = 0.2
beta = 0.1
#
Rth_xN = 0.2
Rth_xF = 0.2

# input array in MATLAB
# sub = [MM; PS_dB; PN_dB; PF_dB; PE_dB; dSN; dSF; dSE; dNE; dFE;]'


for ii in range(len(PS_dB)):
    xx = [MM, PS_dB[ii], PN_dB, PF_dB, PE_dB, dSN, dSE, dNE, dFE, thetaN, Rth_xN, Rth_xF]
    x_test = np.array(xx).reshape(1,-1)
    y_pred = new_model.predict(x_test)
    print(y_pred)
