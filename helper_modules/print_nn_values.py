# -*- coding: utf-8 -*-
"""
Created on Fri May 14 11:59:01 2021

@author: vtac
"""

# Description: Print normalization constants, weights and biases to copy to abaqus.inp file.

starting_index = 0 #in .inp file if the normalization constants start at a new line then
                   #starting_index=0, otherwise specify where it starts.
norm_const = np.array([meanI1, stdI1, meanI2, stdI2, meanI4a, stdI4a, meanI4s, stdI4s, meanPsi, stdPsi])
norm_const = norm_const.astype('float32')
# A = model weights and biases
A = model.get_weights()
n_weights = 88
n_bias = 17 
# weights = np.zeros(n_weights)
# biases = np.zeros(n_bias)
weights = []
biases = []
for i, a in enumerate(A):
    if i % 2 == 0:
        for a2 in a:
            for a3 in a2:
                weights.append(a3)
    else:
        for a2 in a:
            biases.append(a2)

print(*norm_const[:8-starting_index], sep=', ')
starting_index2 = starting_index + 10 - 8
if len(norm_const[8-starting_index:]) > 8:
    print(*norm_const[8-starting_index:8-starting_index+8], sep=', ')
    print(*norm_const[8-starting_index+8:], *weights[:8-starting_index2], sep=', ')
else:
    print(*norm_const[8-starting_index:], *weights[:8-starting_index2], sep=', ')
starting_index = starting_index2
for i in range(int((len(weights)+len(biases))/8)):
    if len(weights) >= 8-starting_index + (i+1)*8:
        print(*weights[8-starting_index + i*8:8-starting_index + (i+1)*8], sep=', ')
    elif len(weights) > 8-starting_index + i*8:
        starting_index2 = int((len(weights)+starting_index) % 8)
        print(*weights[8-starting_index + i*8:], *biases[:8-starting_index2], sep=', ')
        # print(biases[:8-startingin_index2])
        j = i
    else:
        print(*biases[8-starting_index2 + (i-j-1)*8:8-starting_index2 + (i-j)*8], sep=', ')
        
    

    
    
    
    