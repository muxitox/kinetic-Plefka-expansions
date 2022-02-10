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
from hidden_kinetic_ising.hidden_kinetic_ising_it3 import HiddenIsing as HI3

reproducible = True
if reproducible:
    seed = [3813]
else:
    seed = np.random.randint(5000, size=1)

rng = np.random.default_rng(seed)

dataset = 1
gradient_mode = 'regular'
max_reps = 4000
eta = 0.01
burn_in = 100
vis_units = 10
save_results = True

filename = 'data/DataSet' + str(dataset) + '.mat'
mat = scipy.io.loadmat(filename)

print(mat.keys())
data = mat['data']
spikes = data[0][0][0]

N = len(spikes)
visible_idx = rng.choice(range(0, N), vis_units)

r = 70
T = int(np.ceil(3600000 / r))
X_full = csr_matrix((N, T))
X = csr_matrix((vis_units, T))
original_neurons = N
for n in range(N):
    ts = spikes[n][0][0]
    # print(len(ts),min(ts),max(ts))
    inds = np.floor(ts / r).astype(int)
    X_full[n, inds] = 1

X = X_full[visible_idx, :]

X = X[:, 500:]

S = X.sum(axis=1)
S = np.asarray(S).reshape(-1).astype(int)
print(sorted(S))
print(min(S))

thS = 60 * 10
thS = 0
print(len(S), sum(S > thS))

N = sum(S > thS)

X = X[S > thS, :]
d = 1
Xd = X[:, 0:-d]
X = X[:, d:]

X = X.todense() * 2 - 1

m = X.mean(axis=1)
m = np.asarray(m).reshape(-1)

iu1 = np.triu_indices(N, 1)
C = X.dot(X.T) / T
D = X[:, 0:-d].dot(X[:, d:].T) / T
# D = X[:,0:-d].dot(X[:,d:].T)/T
C -= np.einsum('i,k->ik', m, m, optimize=True)
D -= np.einsum('i,k->ik', m, m, optimize=True)

C[range(N), range(N)] = 1 - m ** 2

X = X.T
original_moments = (m, C, D)
b_units_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

T_full, _ = X.shape
T_ori = 1000
T_sim = 3000

X = X[:T_ori, :]

###########
# Learn it2
###########
print('Learning it2 with b_sizes', b_units_list)
# for b_size in b_units_list:
#
#     print('b_size', b_size)
#
#     hidden_ising = HI2(visible_size=vis_units, rng=rng)
#     hidden_ising.set_hidden_size(b_size=b_size)
#     hidden_ising.random_wiring()
#
#     ell_list, error_list, MSE_m_list, MSE_C_list, MSE_D_list, error_iter_list = \
#         hidden_ising.fit(X, eta, max_reps, T_ori, T_sim, original_moments, burn_in=burn_in, gradient_mode=gradient_mode)
#
#     title_str = f'Original size: {original_neurons}. Visible units: {vis_units}. Hidden units: {b_size}.' \
#                 f' O. Simulation steps: {T_ori}. F. Simulation steps: {T_sim}. eta: {eta}. max_reps: {max_reps} '
#     print(title_str)
#
#     num_simulations = 3
#
#     f_MSE_m = 0
#     f_MSE_C = 0
#     f_MSE_D = 0
#
#     # Repeat the simulations to have a good estimation of the error
#     for i in range(0, num_simulations):
#         sim_s = hidden_ising.simulate_hidden(T_sim, burn_in=burn_in)
#         f_sim_m, f_sim_C, f_sim_D = hidden_ising.compute_moments(sim_s, T_sim)
#
#         f_MSE_m += np.mean((m - f_sim_m) ** 2)
#         f_MSE_C += np.mean((C - f_sim_C) ** 2)
#         f_MSE_D += np.mean((D - f_sim_D) ** 2)
#
#     f_MSE_m /= num_simulations
#     f_MSE_C /= num_simulations
#     f_MSE_D /= num_simulations
#
#     MSE_m_list.append(f_MSE_m)
#     MSE_C_list.append(f_MSE_C)
#     MSE_D_list.append(f_MSE_D)
#     error_iter_list.append(max_reps)
#
#     print('Final MSE m', f_MSE_m, 'C', f_MSE_C, 'D', f_MSE_D)
#     print('Final Log-Likelihood', ell_list[-1])
#
#     print()
#
#     fig, ax = plt.subplots(2, figsize=(16, 10), dpi=100)
#     ax[0].plot(ell_list, label='log(ell)')
#     ax[0].plot(np.square(error_list), label='max_grad^2')
#     ax[0].plot(np.diff(ell_list / eta), '--', label='np.diff(log_ell)/eta')
#     ax[0].set_xlabel('iters')
#     ax[0].legend()
#
#     ax[1].plot(error_iter_list, MSE_m_list, label='MSE m')
#     ax[1].plot(error_iter_list, MSE_C_list, label='MSE C')
#     ax[1].plot(error_iter_list, MSE_D_list, label='MSE D')
#     ax[1].set_xlabel('iters')
#     ax[1].legend()
#
#     fig.suptitle(title_str)
#
#     path = f'../hidden_kinetic_ising/results/neurons/it2/{dataset}/{vis_units}/'
#
#     # Check whether the specified path exists or not
#     isExist = os.path.exists(path)
#
#     if not isExist:
#         # Create a new directory because it does not exist
#         os.makedirs(path)
#         print(f"The new directory \"{path} \" is created!")
#
#     eta_str = str(eta).replace('.', '')
#     filename = f"{dataset}_{vis_units}_{b_size}_{T_ori}_{T_sim}_eta{eta_str}_{max_reps}_{burn_in}"
#     if save_results:
#         plt.savefig(path + filename)
#
#         np.savez_compressed(path + filename + '.npz',
#                             H=hidden_ising.H,
#                             J=hidden_ising.J,
#                             M=hidden_ising.M,
#                             K=hidden_ising.K,
#                             L=hidden_ising.L,
#                             b0=hidden_ising.b_0,
#                             m=m,
#                             C=C,
#                             D=D,
#                             MSE_m=f_MSE_m,
#                             MSE_C=f_MSE_C,
#                             MSE_D=f_MSE_D,
#                             log_ell=ell_list[-1])
#     else:
#         plt.show()

###########
# Learn it3
###########
print('Learning it3')

hidden_ising = HI3(visible_size=vis_units, rng=rng)
hidden_ising.random_wiring()

ell_list, error_list, MSE_m_list, MSE_C_list, MSE_D_list, error_iter_list = \
    hidden_ising.fit(X, eta, max_reps, T_ori, T_sim, original_moments, burn_in=burn_in, gradient_mode=gradient_mode)

title_str = f'Seed: {seed}. Original size: {original_neurons}. Visible units: {vis_units}.' \
            f' O. Simulation steps: {T_ori}. F. Simulation steps: {T_sim}. eta: {eta}. max_reps: {max_reps} '
print(title_str)

num_simulations = 3

f_MSE_m = 0
f_MSE_C = 0
f_MSE_D = 0

# Repeat the simulations to have a good estimation of the error
for i in range(0, num_simulations):
    sim_s = hidden_ising.simulate_hidden(T_sim, burn_in=burn_in)
    f_sim_m, f_sim_C, f_sim_D = hidden_ising.compute_moments(sim_s, T_sim)

    f_MSE_m += np.mean((m - f_sim_m) ** 2)
    f_MSE_C += np.mean((C - f_sim_C) ** 2)
    f_MSE_D += np.mean((D - f_sim_D) ** 2)

f_MSE_m /= num_simulations
f_MSE_C /= num_simulations
f_MSE_D /= num_simulations

MSE_m_list.append(f_MSE_m)
MSE_C_list.append(f_MSE_C)
MSE_D_list.append(f_MSE_D)
error_iter_list.append(max_reps)

print('Final MSE m', f_MSE_m, 'C', f_MSE_C, 'D', f_MSE_D)
print('Final Log-Likelihood', ell_list[-1])

print()

fig, ax = plt.subplots(2, figsize=(16, 10), dpi=100)
ax[0].plot(ell_list, label='log(ell)')
ax[0].plot(np.square(error_list), label='max_grad^2')
ax[0].plot(np.diff(ell_list / eta), '--', label='np.diff(log_ell)/eta')
ax[0].set_xlabel('iters')
ax[0].legend()

ax[1].plot(error_iter_list, MSE_m_list, label='MSE m')
ax[1].plot(error_iter_list, MSE_C_list, label='MSE C')
ax[1].plot(error_iter_list, MSE_D_list, label='MSE D')
ax[1].set_xlabel('iters')
ax[1].legend()

fig.suptitle(title_str)

path = f'../hidden_kinetic_ising/results/neurons/it3/{dataset}/{vis_units}/'

# Check whether the specified path exists or not
isExist = os.path.exists(path)

if not isExist:
    # Create a new directory because it does not exist
    os.makedirs(path)
    print(f"The new directory \"{path} \" is created!")

eta_str = str(eta).replace('.', '')
filename = f"{dataset}_{vis_units}_{T_ori}_{T_sim}_eta{eta_str}_{max_reps}_{burn_in}"
if save_results:
    plt.savefig(path + filename)

    np.savez_compressed(path + filename + '.npz',
                        H=hidden_ising.H,
                        J=hidden_ising.J,
                        K=hidden_ising.K,
                        L=hidden_ising.L,
                        b0=hidden_ising.b_0,
                        h0=hidden_ising.h_0,
                        m=m,
                        C=C,
                        D=D,
                        MSE_m=f_MSE_m,
                        MSE_C=f_MSE_C,
                        MSE_D=f_MSE_D,
                        log_ell=ell_list[-1])
else:
    plt.show()
