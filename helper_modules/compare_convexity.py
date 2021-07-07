# -*- coding: utf-8 -*-
"""
Created on Tue May 18 22:16:23 2021

@author: vtac
"""

#Compare convexity of NNs
import pickle
import numpy as np
import matplotlib.pyplot as plt

model_name = 'P1C1_s'

fsize=10
pltparams = {'legend.fontsize': 'large',
          'figure.figsize': (10,5),
          'axes.labelsize': fsize*1.5,
          'axes.titlesize': fsize*1.5,
          'xtick.labelsize': fsize*1.25,
          'ytick.labelsize': fsize*1.25,
          'axes.titlepad': 25,
          "mathtext.fontset": 'dejavuserif'}
plt.rcParams.update(pltparams)
plt.rcParams['axes.prop_cycle'] = plt.cycler('color', plt.cm.inferno(np.linspace(0.2, 0.8,2)))
plt.rcParams["mathtext.fontset"] = 'dejavuserif'

with open('savednet/' + model_name + '_history.pkl', 'rb') as f:
    conv_history = pickle.load(f)
with open('savednet/' + model_name + '_history_nonconv.pkl', 'rb') as f:
    nonconv_history = pickle.load(f)

fig = plt.figure()
gs = fig.add_gridspec(1,2, wspace=0.3, hspace=0.4) #nrows, ncols

ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(   conv_history['sigma_loss'], label='Convex Model')
ax1.plot(nonconv_history['sigma_loss'], label='Non-Convex Model')
ax1.legend()
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Loss on $\sigma$')

symmetry = np.array(conv_history['symmetry'])
conv1 = np.array(conv_history['convexity'])
conv_convexity = symmetry + 10*conv1
symmetry = np.array(nonconv_history['symmetry'])
conv1 = np.array(nonconv_history['convexity'])
nonconv_convexity = symmetry + 10*conv1

ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(conv_convexity, label='Convex Model')
ax2.plot(nonconv_convexity, label='Non-Convex Model')
ax2.set_xlabel('Iteration')
ax2.set_ylabel('Convexity')
