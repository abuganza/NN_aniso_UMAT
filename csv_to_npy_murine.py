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

#Murine, Subject111, Sample 1, in vitro unloaded
offx_path = 'training_data/murine_subject111_invitrounloaded/Subject111_Sample1_YoungDorsal_OffbiaxialX.csv'
offy_path = 'training_data/murine_subject111_invitrounloaded/Subject111_Sample1_YoungDorsal_OffbiaxialY.csv'
equi_path = 'training_data/murine_subject111_invitrounloaded/Subject111_Sample1_YoungDorsal_Equibiaxial.csv'
save_path = 'training_data/S111S1_xyb.npy'
save_path_aug = 'training_data/S111S1_xybs.npy'
#GOH_params = [3.17127895e-03, 1.75806384e-02, 1.14908033e+01, 3.33333333e-01, 5.29525165e-03] #Fitted to offx, offy, equi
GOH_params = [2.69804100e-03, 2.13677988e-02, 1.14909551e+01, 3.33333333e-01, 5.29620414e-03] #Fitted to offx and offy only

#Just for testing, temporary:
#Murine, Subject111, Sample 1, in vitro preloaded
# offx_path = 'training_data/murine_subject111_invitropreloaded/Subject111_Sample1_YoungDorsal_OffbiaxialX.csv'
# offy_path = 'training_data/murine_subject111_invitropreloaded/Subject111_Sample1_YoungDorsal_OffbiaxialY.csv'
# equi_path = 'training_data/murine_subject111_invitropreloaded/Subject111_Sample1_YoungDorsal_Equibiaxial.csv'
# save_path = 'training_data/S111S1pre_xyb.npy'
# save_path_aug = 'training_data/S111S1pre_xybs.npy'
# GOH_params = [5.29185819e-03, 3.04604196e-01, 1.14920204e+01, 3.22288712e-01, 5.54628463e-03] #Fitted to offx and offy only

lambda_max = 1.45
data_offx = np.genfromtxt(offx_path,delimiter=',')
data_offy = np.genfromtxt(offy_path,delimiter=',')
data_equi = np.genfromtxt(equi_path,delimiter=',')
data = np.vstack((data_offx, data_offy, data_equi))


X = np.vstack((data[:,0], data[:,1]))
X = np.transpose(X,[1,0])
Y = np.vstack((data[:,2], data[:,3]))
Y = np.transpose(Y,[1,0])

#The stresses in the csv files already are cauchy stress, no need to convert.
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

with open(save_path_aug,'wb') as f:
    np.save(f,[X_aug, Y_aug])


