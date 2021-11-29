#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt
from plefka_expansions import mf_ising
from plefka_functions import update_m_P_CMS, update_C_P_CMS

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

T = 100
m_mean = np.zeros(T)
C_mean = np.zeros(T)
D_mean = np.zeros(T)

iu1 = np.triu_indices(N, 1)
print('gamma', gamma)
for t in range(T):
    m_p = m.copy()
    C_p = C.copy()
    D_p = D.copy()
    m1, D1 = update_m_P_CMS(H, J, m_p, C_p)
    C1 = update_C_P_CMS(H, J, m_p, m, D1)

    m = gamma * m1.copy() + (1 - gamma) * m_p
    D = (C_p + np.einsum('i,l->il', m_p, m_p, optimize=True)) * (1 - gamma) + (
                D1 + np.einsum('i,l->il', m1, m_p, optimize=True)) * gamma \
        - np.einsum('i,l->il', m, m_p, optimize=True)
    C = (C_p + np.einsum('i,l->il', m_p, m_p, optimize=True)) * (1 - gamma) ** 2 + gamma * (1 - gamma) * (
                D1 + np.einsum('i,l->il', m1, m_p, optimize=True)
                + D1.T + np.einsum('i,l->li', m1, m_p, optimize=True)) + gamma ** 2 * (
                    C1 + np.einsum('i,l->il', m1, m1, optimize=True)) - np.einsum('i,l->il', m, m, optimize=True)
    C[range(N), range(N)] = 1 - m ** 2

    #    I.D = D1.copy()
    #    I.C = C1.copy()

    m_mean[t] = np.mean(m)
    C_mean[t] = np.mean(C)
    D_mean[t] = np.mean(D)
    print(t)
    print(m_mean[t], C_mean[t], D_mean[t])
    print(np.mean(mexp), np.mean(Cexp[iu1]), np.mean(Dexp))

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
