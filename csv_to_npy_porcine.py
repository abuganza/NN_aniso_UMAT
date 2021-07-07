#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 22:21:00 2021

@author: vahidullahtac
Description: Take csv files of P1C1 and save the collected data as npy
"""

import numpy as np
from material_models import GOH_fullyinc

# %% I. Pure Unadulterated Experimental Data

#Porcine, P1C1
# offx_path = 'training_data/porcine_P1C1/P1C1S1_OffX.csv'
# offy_path = 'training_data/porcine_P1C1/P1C1S1_OffY.csv'
# save_path = 'training_data/P1C1_xy.npy'
# save_path_aug = 'training_data/P1C1_xys.npy'
# save_path_synth = 'training_data/P1C1_s.npy'
# GOH_params = [9.86876414e-04, 5.64353050e-01, 7.95242698e+01, 2.94747207e-01, 1.57079633e+00]
# # GOH_params = [1.02356332e-02, 5.13664702e-01, 5.91491834e+01, 2.74447648e-01, 1.57079633e+00]
# lambda_max = 1.25
# data_offx = np.genfromtxt(offx_path,delimiter=',')
# data_offy = np.genfromtxt(offy_path,delimiter=',')
# data = np.vstack((data_offx, data_offy))

#Porcine, P2C1
offx_path = 'training_data/porcine_P2C1/P2C1S1_OffX.csv'
offy_path = 'training_data/porcine_P2C1/P2C1S1_OffY.csv'
save_path = 'training_data/P2C1_xy.npy'
save_path_aug = 'training_data/P2C1_xys.npy'
save_path_synth = 'training_data/P2C1_s.npy'
GOH_params = [8.41516969e-03, 1.41086463e+00, 7.95267699e+01, 3.08747892e-01, 1.57079633e+00]
# GOH_params = [1.02356332e-02, 5.13664702e-01, 5.91491834e+01, 2.74447648e-01, 1.57079633e+00]
lambda_max = 1.25
data_offx = np.genfromtxt(offx_path,delimiter=',')
data_offy = np.genfromtxt(offy_path,delimiter=',')
data = np.vstack((data_offx, data_offy))


X = np.vstack((data[:,0], data[:,2]))
X = np.transpose(X,[1,0])
Y = np.vstack((data[:,1], data[:,3]))
Y = np.transpose(Y,[1,0])

#The stresses in the csv files are PK1 stress. Convert it to Cauchy stress
F = np.zeros([X.shape[0],3,3])
F[:,0,0] = X[:,0]
F[:,1,1] = X[:,1]
F[:,2,2] = 1/(X[:,0]*X[:,1])
P = np.zeros_like(F)
P[:,0,0] = Y[:,0]
P[:,1,1] = Y[:,1]
sigma = P*F #Since F_T=F
Y = np.zeros_like(X)
Y[:,0] = sigma[:,0,0]
Y[:,1] = sigma[:,1,1]

with open(save_path, 'wb') as f:
    np.save(f,[X,Y])

# %% II. Fill up the input space with synthetic data to guide the neural network in the right direction
res1 = 11 #Resolution of data points (lambda_1)
res2 = 4 #Resolution of lambda_2 when lambda_1 = 1
res3 = res1 #Resolution of lambda_2 when lambda_1 varies
lambda1 = 1.0
X_synth = np.ones([res1*(res2+res3),2])
Y_synth = np.zeros([res1*(res2+res3),2])
for i in range(res1):
    X_synth[i*(res2 + res3)       : i*(res2 + res3) +          res2,0] = np.linspace(1.0,lambda1,res2) 
    X_synth[i*(res2 + res3) + res2: i*(res2 + res3) + (res2 + res3),0] = [lambda1]*res3
    X_synth[i*(res2 + res3) + res2: i*(res2 + res3) + (res2 + res3),1] = np.linspace(1.0, lambda_max, res3)
    lambda1+= (lambda_max-1.0)/res3

GOH = GOH_fullyinc(X_synth, GOH_params[0], GOH_params[1], GOH_params[2], GOH_params[3], GOH_params[4])
Y_synth[:,0] = GOH.s(X_synth)[:,0,0]
Y_synth[:,1] = GOH.s(X_synth)[:,1,1]

X_aug = np.vstack((X,X_synth))
Y_aug = np.vstack((Y,Y_synth))

with open(save_path_synth,'wb') as f:
    np.save(f,[X_synth, Y_synth])
    
with open(save_path_aug,'wb') as f:
    np.save(f,[X_aug, Y_aug])


