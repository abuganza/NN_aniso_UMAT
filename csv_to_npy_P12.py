#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 22:21:00 2021

@author: vahidullahtac
Description: Take csv files of P1C1 and save the collected data as npy
"""

import numpy as np
from material_models import GOH_fullyinc

#Porcine, P12AC1
# offx_path = 'training_data/porcine_P12AC1/P12AC1S1_OffX.csv'
# offy_path = 'training_data/porcine_P12AC1/P12AC1S1_OffY.csv'
# equi_path = 'training_data/porcine_P12AC1/P12AC1S1_Equibiaxial.csv'
# strx_path = 'training_data/porcine_P12AC1/P12AC1S1_StripX.csv'
# stry_path = 'training_data/porcine_P12AC1/P12AC1S1_StripY.csv'
# save_path_1  = 'training_data/P12AC1_xy.npy' #x: Offx, y: Offy, b: Biaxial, s: synthetic, sx: Stripx, sy: Stripy
# save_path_2  = 'training_data/P12AC1_xys.npy'
# save_path_3  = 'training_data/P12AC1_xyb.npy'
# save_path_4  = 'training_data/P12AC1_xybs.npy'
# save_path_5  = 'training_data/P12AC1_xysxsy.npy'
# save_path_6  = 'training_data/P12AC1_xysxsys.npy'
# save_path_7  = 'training_data/P12AC1_sxsy.npy'
# save_path_8  = 'training_data/P12AC1_sxsys.npy'
# save_path_9  = 'training_data/P12AC1_bsxsy.npy'
# save_path_10 = 'training_data/P12AC1_bsxsys.npy'
# save_path_11 = 'training_data/P12AC1_xybsxsy.npy'
# GOH_xy     = [ 0.,          9.24894751,  3.36772826,  0.30367759,  1.57094141] #Offx + Offy
# GOH_xyb    = [ 0.,          7.70454001, 81.45413033,  0.30407081,  1.57111209] #Offx + Offy + Equi
# GOH_xysxsy = [ 0.,         10.0699847 ,  2.92145503,  0.30207647,  1.57084425] #Offx + Offy + Strx + Stry
# GOH_sxsy   = [ 0.,         12.32983437,  2.99638755,  0.29977702,  1.57084442] #Strx + Stry
# GOH_bsxsy  = [ 0.,         11.87718311,  4.14343856,  0.30260715,  1.570626  ] #Equi + Strx + Stry
# lambda_max = 1.25
# data_offx = np.genfromtxt(offx_path,delimiter=',')
# data_offx = data_offx[:-3] #Remove bad data
# data_offy = np.genfromtxt(offy_path,delimiter=',')
# data_equi = np.genfromtxt(equi_path,delimiter=',')
# data_equi = data_equi[:-20]
# data_strx = np.genfromtxt(strx_path,delimiter=',')
# data_stry = np.genfromtxt(stry_path,delimiter=',')
# data_stry = data_stry[:-29]

#Porcine, P12BC2
offx_path = 'training_data/porcine_P12BC2/P12BC2S1_OffX.csv'
offy_path = 'training_data/porcine_P12BC2/P12BC2S1_OffY.csv'
equi_path = 'training_data/porcine_P12BC2/P12BC2S1_Equibiaxial.csv'
strx_path = 'training_data/porcine_P12BC2/P12BC2S1_StripX.csv'
stry_path = 'training_data/porcine_P12BC2/P12BC2S1_StripY.csv'
save_path_1  = 'training_data/P12BC2_xy.npy' #x: Offx, y: Offy, b: Biaxial, s: synthetic, sx: Stripx, sy: Stripy
save_path_2  = 'training_data/P12BC2_xys.npy'
save_path_3  = 'training_data/P12BC2_xyb.npy'
save_path_4  = 'training_data/P12BC2_xybs.npy'
save_path_5  = 'training_data/P12BC2_xysxsy.npy'
save_path_6  = 'training_data/P12BC2_xysxsys.npy'
save_path_7  = 'training_data/P12BC2_sxsy.npy'
save_path_8  = 'training_data/P12BC2_sxsys.npy'
save_path_9  = 'training_data/P12BC2_bsxsy.npy'
save_path_10 = 'training_data/P12BC2_bsxsys.npy'
save_path_11 = 'training_data/P12BC2_xybsxsy.npy'
GOH_xy     = [0.,         6.66375239, 2.10007379, 0.32194361, 1.57089277] #Offx + Offy
GOH_xyb    = [0.,         6.45220922, 2.24243604, 0.31306001, 1.57132957] #Offx + Offy + Equi
GOH_xysxsy = [0.,         5.40692214, 1.58278121, 0.30029796, 1.57079144] #Offx + Offy + Strx + Stry
GOH_sxsy   = [0.,         5.56814876, 2.17702876, 0.28625754, 1.57090147] #Strx + Stry
GOH_bsxsy  = [0.,         5.28151486, 2.0823615 , 0.2881214 , 1.57106945] #Equi + Strx + Stry
lambda_max = 1.25
data_offx = np.genfromtxt(offx_path,delimiter=',')
data_offy = np.genfromtxt(offy_path,delimiter=',')
data_equi = np.genfromtxt(equi_path,delimiter=',')
data_equi = data_equi[:-6]
data_strx = np.genfromtxt(strx_path,delimiter=',')
data_stry = np.genfromtxt(stry_path,delimiter=',')
data_stry = data_stry[:-17]

# %% 1. Save Offx + Offy data only
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

with open(save_path_1, 'wb') as f:
    np.save(f,[X,Y])

#### Fill up the input space with synthetic data to guide the neural network in the right direction
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

GOH = GOH_fullyinc(X_synth, GOH_xy[0], GOH_xy[1], GOH_xy[2], GOH_xy[3], GOH_xy[4])
Y_synth[:,0] = GOH.s(X_synth)[:,0,0]
Y_synth[:,1] = GOH.s(X_synth)[:,1,1]

X_aug = np.vstack((X,X_synth))
Y_aug = np.vstack((Y,Y_synth))
    
with open(save_path_2,'wb') as f:
    np.save(f,[X_aug, Y_aug])

# %% 2. Save Offx + Offy + Equibiaxial
data = np.vstack((data_offx, data_offy, data_equi))

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

with open(save_path_3, 'wb') as f:
    np.save(f,[X,Y])

#### Fill up the input space with synthetic data to guide the neural network in the right direction
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

GOH = GOH_fullyinc(X_synth, GOH_xyb[0], GOH_xyb[1], GOH_xyb[2], GOH_xyb[3], GOH_xyb[4])
Y_synth[:,0] = GOH.s(X_synth)[:,0,0]
Y_synth[:,1] = GOH.s(X_synth)[:,1,1]

X_aug = np.vstack((X,X_synth))
Y_aug = np.vstack((Y,Y_synth))
    
with open(save_path_4,'wb') as f:
    np.save(f,[X_aug, Y_aug])


# %% 3. Save Offx + Offy + Stripx + Stripy
data = np.vstack((data_offx, data_offy, data_strx, data_stry))

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

with open(save_path_5, 'wb') as f:
    np.save(f,[X,Y])

#### Fill up the input space with synthetic data to guide the neural network in the right direction
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

GOH = GOH_fullyinc(X_synth, GOH_xysxsy[0], GOH_xysxsy[1], GOH_xysxsy[2], GOH_xysxsy[3], GOH_xysxsy[4])
Y_synth[:,0] = GOH.s(X_synth)[:,0,0]
Y_synth[:,1] = GOH.s(X_synth)[:,1,1]

X_aug = np.vstack((X,X_synth))
Y_aug = np.vstack((Y,Y_synth))
    
with open(save_path_6,'wb') as f:
    np.save(f,[X_aug, Y_aug])


# %% 4. Save Stripx + Stripy
data = np.vstack((data_strx, data_stry))

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

with open(save_path_7, 'wb') as f:
    np.save(f,[X,Y])

#### Fill up the input space with synthetic data to guide the neural network in the right direction
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

GOH = GOH_fullyinc(X_synth, GOH_xysxsy[0], GOH_xysxsy[1], GOH_xysxsy[2], GOH_xysxsy[3], GOH_xysxsy[4])
Y_synth[:,0] = GOH.s(X_synth)[:,0,0]
Y_synth[:,1] = GOH.s(X_synth)[:,1,1]

X_aug = np.vstack((X,X_synth))
Y_aug = np.vstack((Y,Y_synth))
    
with open(save_path_8,'wb') as f:
    np.save(f,[X_aug, Y_aug])

# %% 5. Save Stripx + Stripy
data = np.vstack((data_equi, data_strx, data_stry))

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

with open(save_path_9, 'wb') as f:
    np.save(f,[X,Y])

#### Fill up the input space with synthetic data to guide the neural network in the right direction
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

GOH = GOH_fullyinc(X_synth, GOH_bsxsy[0], GOH_bsxsy[1], GOH_bsxsy[2], GOH_bsxsy[3], GOH_bsxsy[4])
Y_synth[:,0] = GOH.s(X_synth)[:,0,0]
Y_synth[:,1] = GOH.s(X_synth)[:,1,1]

X_aug = np.vstack((X,X_synth))
Y_aug = np.vstack((Y,Y_synth))
    
with open(save_path_10,'wb') as f:
    np.save(f,[X_aug, Y_aug])

# %% 6. Save all experimental data
data = np.vstack((data_offx, data_offy, data_equi, data_strx, data_stry))

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

with open(save_path_11, 'wb') as f:
    np.save(f,[X,Y])