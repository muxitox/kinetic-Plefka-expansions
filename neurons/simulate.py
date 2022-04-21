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

import numpy as np
from matplotlib import pyplot as plt
from plefka.plefka_functions import update_m_P_CMS, update_C_P_CMS
import os

def update_moments_gamma(H, J, N, m_p, C_p, gamma):

    m1, D1 = update_m_P_CMS(H, J, m_p, C_p)
    C1 = update_C_P_CMS(H, J, m_p, m1, D1)

    m = gamma * m1.copy() + (1 - gamma) * m_p
    D = (C_p + np.einsum('i,l->il', m_p, m_p, optimize=True)) * (1 - gamma) + (
            D1 + np.einsum('i,l->il', m1, m_p, optimize=True)) * gamma \
        - np.einsum('i,l->il', m, m_p, optimize=True)
    C = (C_p + np.einsum('i,l->il', m_p, m_p, optimize=True)) * (1 - gamma) ** 2 + gamma * (1 - gamma) * (
            D1 + np.einsum('i,l->il', m1, m_p, optimize=True)
            + D1.T + np.einsum('i,l->li', m1, m_p, optimize=True)) + gamma ** 2 * (
                C1 + np.einsum('i,l->il', m1, m1, optimize=True)) - np.einsum('i,l->il', m, m, optimize=True)
    C[range(N), range(N)] = 1 - m ** 2

    return m, C, D


if __name__ == "__main__":
    plt.rc('text', usetex=True)
    font = {'size': 15}
    plt.rc('font', **font)
    plt.rc('legend', **{'fontsize': 12})

    dataset = 1
    filename = 'stats/stats' + str(dataset) + '.npz'
    data = np.load(filename)
    N = data['N']
    Cexp = data['C']
    Dexp = data['D']
    mexp = data['m']

    filename = 'networks/net_' + str(dataset) + '.npz'
    data = np.load(filename)
    N = data['N']
    H = data['H']
    J = data['J']
    gamma = data['gamma']

    ##I = mf_ising(N)
    # I.J=J
    # I.H=H
    # I.m=mexp.copy()
    # I.C=Cexp.copy()
    # I.D=Cexp.copy()
    m = mexp.copy()
    C = Cexp.copy() * 0
    D = Dexp.copy() * 0

    T = 128
    m_mean = np.zeros(T)
    C_mean = np.zeros(T)
    D_mean = np.zeros(T)

    iu1 = np.triu_indices(N, 1)
    print('gamma', gamma)
    for t in range(T):
        m_p = m.copy()
        C_p = C.copy()
        D_p = D.copy()

        m, C, D = update_moments_gamma(H, J, N, m_p, C_p, gamma)

        m_mean[t] = np.mean(m)
        C_mean[t] = np.mean(C)
        D_mean[t] = np.mean(D)
        print(t)
        print(m_mean[t], C_mean[t], D_mean[t])
        print(np.mean(mexp), np.mean(Cexp[iu1]), np.mean(Dexp))


    folder = 'results/moments'
    isExist = os.path.exists(folder)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(folder)
    filename = folder + '/mom_' + str(dataset) + '.npz'
    np.savez_compressed(filename,
                        mean_m_t=m_mean,
                        mean_C_t=C_mean,
                        mean_D_t=D_mean,
                        final_m_t=m,
                        final_C_t=C,
                        final_D_t=D,
                        T=128
                        )

    plt.figure()
    plt.plot([min(mexp), max(mexp)], [min(mexp), max(mexp)], 'k')
    plt.plot(mexp, m, '.')
    plt.figure()
    plt.plot([min(Cexp[iu1]), max(Cexp[iu1])], [min(Cexp[iu1]), max(Cexp[iu1])], 'k')
    plt.plot(Cexp[iu1], C[iu1], '.')
    plt.figure()
    plt.plot([np.min(Dexp), np.max(Dexp)], [np.min(Dexp), np.max(Dexp)], 'k')
    plt.plot(Dexp.flatten(), D.flatten(), '.')
    plt.figure()
    plt.plot(m_mean)
    plt.figure()
    plt.plot(C_mean)
    plt.plot(D_mean)
    plt.show()
