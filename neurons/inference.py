#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 15:45:08 2018

@author: maguilera
"""

from mf_ising import mf_ising
from plefka_functions import update_m_P_CMS, update_C_P_CMS
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm

from scipy.optimize import minimize

plt.rc('text', usetex=True)
font = {'size': 15}
plt.rc('font', **font)
plt.rc('legend', **{'fontsize': 12})


def nsf(num, n=4):
    """n-Significant Figures"""
    numstr = ("{0:.%ie}" % (n - 1)).format(num)
    return float(numstr)


dataset = 1
filename = 'stats/stats' + str(dataset) + '.npz'
data = np.load(filename)
N = data['N']
Cexp = data['C']
Dexp = data['D']
mexp = data['m']

print(mexp.shape, Cexp.shape, Dexp.shape)
iu1 = np.triu_indices(N, 1)
plt.figure()
plt.hist(Cexp[iu1], 100)
plt.figure()
plt.hist(Dexp[iu1], 100)
# plt.show()


eta = 1
error_ref = 0.0001
max_rep = 5000
# error_ref=0.05

etaH = eta
# etaJ=eta/N*5
etaJ = eta / N ** 0.5
etaJ1 = eta / 10
# etaJ=eta/10


I = mf_ising(N)
error = 1
min_error = error
H = np.arctanh(mexp)
J = np.zeros((N, N))
rep = 0
rep_min = 0

gamma = 0.77
# gamma=1
Dexp1 = (Dexp + np.einsum('i,l->il', mexp, mexp, optimize=True)) / gamma - (1 - gamma) / gamma * (
            Cexp + np.einsum('i,l->il', mexp, mexp, optimize=True)) - np.einsum('i,l->il', mexp, mexp, optimize=True)

while error > error_ref:

    #	I.m,_,I.D = update_P1D_o2(I.H, I.J,mexp,Cexp, Dexp)
    #	I.update_P1_o2()
    #	I.update_P1C_o2()
    #	m1 = update_m_P_t_o2(H, J, mexp,Cexp)
    #	D1 = update_D_P_t_o2(H, J, m1,mexp,Cexp)
    #	C1 = update_C_P_t_o2(H, J, m1,Cexp)

    m1, D1 = update_m_P_CMS(H, J, mexp, Cexp)
    Cn = update_C_P_CMS(H, J, mexp, m1, D1)
    C1 = (1 - gamma) ** 2 * (Cexp + np.einsum('i,l->il', mexp, mexp, optimize=True)) + gamma * (1 - gamma) * (
                D1 + np.einsum('i,l->il', m1, mexp, optimize=True) + D1.T + np.einsum('i,l->li', m1, mexp,
                                                                                      optimize=True)) + gamma ** 2 * (
                     Cn + np.einsum('i,l->il', m1, m1, optimize=True)) - np.einsum('i,l->il', m1, m1, optimize=True)
    C1[range(N), range(N)] = 1 - m1 ** 2

    DJ = Dexp1 - D1
    DH = mexp - m1

    error = max(np.max(np.abs(DH)), np.max(np.abs(DJ)))

    J += etaJ * DJ  # - 0.000*I.J
    J[range(N), range(N)] += (etaJ1 - etaJ) * np.diag(DJ)
    #	np.clip(I.J,-1/N/10,1/N/10)
    #	if np.max(np.abs(DH))>np.max(np.abs(DJ)):
    H += etaH * DH
    if rep % 10 == 0:
        print('P_o2', rep, np.max(np.abs(DH)), np.max(np.abs(DJ)))
        print('P_o2', 'moments', nsf(np.mean(mexp)), nsf(np.mean(Dexp1[iu1])), nsf(np.mean(np.diag(Dexp1))), '|',
              nsf(np.mean(m1)), nsf(np.mean(D1[iu1])), nsf(np.mean(np.diag(D1))))
        print(nsf(np.mean(Cexp[iu1])), nsf(np.mean(C1[iu1])))
        print(np.mean((m1 - mexp)) ** 2, np.mean((D1 - Dexp1) ** 2), np.mean((C1 - Cexp) ** 2))

    rep += 1
    rep_min += 1
    if error < error_ref or rep > max_rep:
        print(error < error_ref, rep > max_rep)
        break

filename = 'networks/net_' + str(dataset) + '.npz'
np.savez_compressed(filename, H=H, J=J, N=N, gamma=gamma)

exit()
# cmap = cm.get_cmap('inferno_r')
# colors=[]
# for i in range(4):
#	colors+=[cmap((i+0.5)/4)]
#	
# plt.figure(figN=(5, 4),dpi=300)
# plt.plot(H,HP0o2, 'v', color=colors[0], ms=5, label='P[t-1:t]', rasterized=True)
# plt.plot(H,HP1o2, 's', color=colors[1], ms=5, label='P[t]', rasterized=True)
# plt.plot(H,HP2o1, 'd', color=colors[2], ms=5, label='P[t-1]', rasterized=True)
# plt.plot(H,HP1Co2, 'o', color=colors[3], ms=5, label='P[D]', rasterized=True)
# plt.plot([np.min(H),np.max(H)],[np.min(H),np.max(H)],'k')
# plt.axis([np.min(H),np.max(H),np.min(np.concatenate((HP0o2,HP1o2,HP1Co2))),np.max(np.concatenate((HP0o2,HP1o2,HP1Co2)))])
# plt.xlabel(r'$H_i^r$', fontN=18)
# plt.ylabel(r'$H_i^m$', fontN=18, rotation=0, labelpad=15)
# plt.title(r'$\beta/\beta_c=' + str(beta) + r'$', fontN=18)
# plt.legend()
# plt.savefig('img/distribution_H-beta_' +
#            str(int(beta * 100)) + '.pdf', bbox_inches='tight')
#            
# plt.figure(figN=(5, 4),dpi=300)
# plt.plot(J.flatten(),JP0o2.flatten(), 'v', color=colors[0], ms=5, label='P[t-1:t]', rasterized=True)
# plt.plot(J.flatten(),JP2o1.flatten(), 'd', color=colors[2], ms=5, label='P[t-1]', rasterized=True)
# plt.plot(J.flatten(),JP1o2.flatten(), 's', color=colors[1], ms=5, label='P[t]', rasterized=True)
# plt.plot(J.flatten(),JP1Co2.flatten(), 'o', color=colors[3], ms=5, label='P[D]', rasterized=True)
# plt.plot([np.min(J),np.max(J)],[np.min(J),np.max(J)],'k')
# plt.axis([np.min(J),np.max(J),np.min(2*np.concatenate((JP0o2,JP1Co2))),2*np.max(np.concatenate((JP0o2,JP1Co2)))])
# plt.xlabel(r'$J_{ij}^r$', fontN=18)
# plt.ylabel(r'$J_{ij}^m$', fontN=18, rotation=0, labelpad=15)
# plt.title(r'$\beta/\beta_c=' + str(beta) + r'$', fontN=18)
# plt.legend()
# plt.savefig('img/distribution_J-beta_' +
#            str(int(beta * 100)) + '.pdf', bbox_inches='tight')


# names = ['P[t-1:t]', 'P[t]', 'P[t-1]','P[D]']

# errorH=np.array([np.mean((H-HP0o2)**2), np.mean((H-HP1o2)**2), np.mean((H-HP1o2)**2), np.mean((H-HP1Co2)**2)])
# print(errorH)
# errorJ=np.array([np.mean((J.flatten()-JP0o2.flatten())**2), np.mean((J.flatten()-JP1o2.flatten())**2), np.mean((J.flatten()-JP2o1.flatten())**2), np.mean((J.flatten()-JP1Co2.flatten())**2)]  )
# print(errorJ)

# filename='img/compare-J_' + str(int(beta * 100)) +'.npz'
# np.savez_compressed(filename,
#      H=H, J=J, 
#      HP0o2=HP0o2, HP1o2=HP1o2, HP2o1= HP2o1, HP1Co2=HP1Co2,
#      JP0o2=JP0o2, JP1o2=JP1o2, JP2o1= JP2o1, JP1Co2=JP1Co2)


# plt.show()
