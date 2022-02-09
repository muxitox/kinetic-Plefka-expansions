#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Wed Dec  5 15:45:08 2018

Hyperparameters for learning dataset 1 from https://crcns.org/data-sets/ssc/ssc-3/about-ssc-3
according to "Inference in neural networks using conditional mean-field methods" (https://arxiv.org/abs/2107.06850)

r(extraction): 70 ms
gamma(inference&simulation): 0.70
eta(inference): 1
T(simulation): 128
max_rep(inference): 2150

Exact hyperparameters are missing due to a data loss resulting from a mechanical failure in a Hard Drive

@author: maguilera and Angel Poc
"""

import scipy.io
import numpy as np
from matplotlib import pyplot as plt
from scipy.sparse import csr_matrix
import os
from hidden_kinetic_ising.hidden_kinetic_ising_it2 import HiddenIsing as HI2


reproducible = True
if reproducible:
    seed = [3813]
else:
    seed = np.random.randint(5000, size=1)

rng = np.random.default_rng(seed)

dataset=1
filename='data/DataSet'+str(dataset)+'.mat'
mat = scipy.io.loadmat(filename)

print(mat.keys())
data=mat['data']
spikes=data[0][0][0]

N=len(spikes)
vis_units = 10
b_size = 1
visible_idx = rng.choice(range(0, N), vis_units)


r=70
T=int(np.ceil(3600000/r))
X_full=csr_matrix((N,T))
X=csr_matrix((vis_units,T))
for n in range(N):
	ts = spikes[n][0][0]
	# print(len(ts),min(ts),max(ts))
	inds = np.floor(ts/r).astype(int)
	X_full[n, inds] = 1

X = X_full[visible_idx, :]

X=X[:,500:]

S=X.sum(axis=1)
S=np.asarray(S).reshape(-1).astype(int)
print(sorted(S))
print(min(S))

thS=60*10
thS = 0
print(len(S),sum(S>thS))

N=sum(S>thS)

X=X[S>thS,:]
d=1
Xd = X[:,0:-d]
X = X[:,d:]

X = X.todense()*2-1

m=X.mean(axis=1)
m=np.asarray(m).reshape(-1)

iu1 = np.triu_indices(N, 1)
C = X.dot(X.T)/T
D = X[:,0:-d].dot(X[:,d:].T)/T
# D = X[:,0:-d].dot(X[:,d:].T)/T
C -= np.einsum('i,k->ik',m,m, optimize=True)
D -= np.einsum('i,k->ik',m,m, optimize=True)

C[range(N),range(N)] = 1 - m**2


hidden_ising = HI2(visible_size=vis_units, rng=rng)
hidden_ising.set_hidden_size(b_size=b_size)
hidden_ising.random_wiring()
