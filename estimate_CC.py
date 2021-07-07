# Author: Vahidullah Tac
# Description: Estimate elasticity tensor, CC, by numerically differentiating the stress tensor

import numpy as np
import pickle
from numpy import linalg as LA
import tensorflow as tf

model_name = 'S111S1_xys'
model = tf.keras.models.model_from_json(open('savednet/' + model_name + '.json').read())
model.load_weights('savednet/' + model_name + '_weights.h5')

with open('savednet/' + model_name + '_factors.pkl','rb') as f:
    norm_factors = pickle.load(f)
_, stdPsi = norm_factors['meanPsi'], norm_factors['stdPsi']
meanI1, stdI1   = norm_factors['meanI1'],  norm_factors['stdI1']
meanI2, stdI2   = norm_factors['meanI2'],  norm_factors['stdI2']
meanI4a, stdI4a = norm_factors['meanI4a'], norm_factors['stdI4a']
meanI4s, stdI4s = norm_factors['meanI4s'], norm_factors['stdI4s']
a0 = np.array([1,0,0])
s0 = np.array([0,1,0])
Kvol = 10

def S_from_NN(C):
    J = np.sqrt(LA.det(C))
    Chat = J**(-2/3)*C
    Cinv = LA.inv(C)
    I1 = Chat.trace()
    I2 = 0.5*(I1**2 - (Chat**2).trace())
    I4a = np.einsum('i,ij,j', a0, Chat, a0)
    I4s = np.einsum('i,ij,j', s0, Chat, s0)
    
    I1norm  = (I1     - meanI1) /stdI1
    I2norm  = (I2     - meanI2) /stdI2
    I4anorm = (I4a    - meanI4a)/stdI4a
    I4snorm = (I4s    - meanI4s)/stdI4s
    
    #### Combine the NN inputs
    inputs = np.zeros([1,4])
    inputs[:,0] = I1norm
    inputs[:,1] = I2norm
    inputs[:,2] = I4anorm
    inputs[:,3] = I4snorm

    inputstensor = tf.Variable(inputs)
    # with tf.GradientTape() as t:
    #     y_pred = model(inputstensor)
    # grad = t.jacobian(y_pred, inputstensor)
    # grad = grad[0,:,0,:]
    # H = grad[1:,:]
    # Psi1  = tf.cast(y_pred[:,1]*stdPsi/stdI1,tf.float64)
    # Psi2  = tf.cast(y_pred[:,2]*stdPsi/stdI2,tf.float64)
    # Psi4a = tf.cast(y_pred[:,3]*stdPsi/stdI4a,tf.float64)
    # Psi4s = tf.cast(y_pred[:,4]*stdPsi/stdI4s,tf.float64)
    # Psi11   = H[0,0]*stdPsi/stdI1**2
    # Psi12   = H[0,1]*stdPsi/stdI1/stdI2
    # Psi14a  = H[0,2]*stdPsi/stdI1/stdI4a
    # Psi14s  = H[0,3]*stdPsi/stdI1/stdI4s
    # Psi22   = H[1,1]*stdPsi/stdI2**2
    # Psi24a  = H[1,2]*stdPsi/stdI2/stdI4a
    # Psi24s  = H[1,3]*stdPsi/stdI2/stdI4s
    # Psi4a4a = H[2,2]*stdPsi/stdI4a**2
    # Psi4a4s = H[2,3]*stdPsi/stdI4a/stdI4s
    # Psi4s4s = H[3,3]*stdPsi/stdI4s**2
    y_pred = model(inputstensor)
    d1 = y_pred[:,1]*stdPsi/stdI1
    d2 = y_pred[:,2]*stdPsi/stdI2
    d3 = y_pred[:,3]*stdPsi/stdI4a
    d4 = y_pred[:,4]*stdPsi/stdI4s
    
    a0a0 = np.outer(a0,a0.T)
    s0s0 = np.outer(s0,s0.T)
    
    # #Method 1: Follow the NN code
    # p = -(2*d1 + 2*d2*(I1-C[2,2]))*C[2,2] #from sigma_3 = 0
    # S = 2*d1*np.eye(3) + 2*d2*(I1*np.eye(3) - Chat) + 2*d3*a0a0 + 2*d4*s0s0 + p*Cinv
    
    #Method 2: Follow the UMAT file
    S_hat = 2*d1*np.eye(3) + 2*d2*(I1*np.eye(3) - Chat) + 2*d3*a0a0 + 2*d4*s0s0
    II = 0.5*(np.einsum('ik,jl->ijkl', np.eye(3), np.eye(3)) + np.einsum('il,jk->ijkl', np.eye(3), np.eye(3)))
    PP1 = II - 1/3*np.einsum('ij,kl->ijkl', Cinv, C)
    S_iso = J**(-2/3)*np.einsum('ijkl,kl->ij', PP1, S_hat)
    p = 2*Kvol*(J-1)
    S_vol = J*p*Cinv
    S = S_iso + S_vol
    return S_hat, S_vol, S_iso, S
def eval_S_hat(C):
    S_hat, S_vol, S_iso, S = S_from_NN(C)
    return S_hat
def eval_S_vol(C):
    S_hat, S_vol, S_iso, S = S_from_NN(C)
    return S_vol
def eval_S_iso(C):
    S_hat, S_vol, S_iso, S = S_from_NN(C)
    return S_iso
def eval_S(C):
    S_hat, S_vol, S_iso, S = S_from_NN(C)
    return S

def eval_dSdC(C, funcS):
    dSdC = np.zeros([3,3,3,3])
    epsilon = 1.e-3
    for i in range(3):
        for j in range(3):
            C_p = np.array(C)
            C_m = np.array(C)
            C_p[i,j]+= epsilon
            C_p[j,i]+= epsilon
            C_m[i,j]-= epsilon
            C_m[j,i]-= epsilon
            
            S_p = funcS(C_p)
            S_m = funcS(C_m)
            
            dSdC[:,:,i,j] = (S_p-S_m)/(4*epsilon)
    return dSdC

F = np.zeros([3,3])
# F = np.eye(3)
F =  np.array([[ 0.970478743248742,      -1.914955218353384E-002,  2.046267640474990E-002],
 [-1.110734136749649E-002,  0.920667023542595,      -7.030317407150114E-003],
 [ 2.933908039611295E-002, -3.390413266609481E-002,   1.11839005271613 ]]).T
C = np.einsum('ij,jk->ik', F.T, F)
Cinv = LA.inv(C)
J = LA.det(F)
Chat = J**(-2/3)*C
# II2 = np.einsum('ik,jl->ijkl', np.eye(3), np.eye(3))
II = 0.5*(np.einsum('ik,jl->ijkl', np.eye(3), np.eye(3)) + np.einsum('il,jk->ijkl', np.eye(3), np.eye(3)))
PP1 = II - 1/3*np.einsum('ij,kl->ijkl', Cinv, C)
dChatdC = J**(-2/3)*np.transpose(PP1, (2,3,0,1))
S     = eval_S(C)
S_vol = eval_S_vol(C)
S_hat = eval_S_hat(C)
S_iso = eval_S_iso(C)

dShatdC = eval_dSdC(C,eval_S_hat)
dSvoldC = eval_dSdC(C,eval_S_vol)
dSisodC = eval_dSdC(C,eval_S_iso)
dSdC    = eval_dSdC(C,eval_S)

#note that CC_hat =/ 2*dShat/dC
CC_vol = 2*dSvoldC
CC_iso = 2*dSisodC
CC     = 2*dSdC
print("S")
print(S)
print("CC")
print(CC)
# print("CC")
# print(CC)


