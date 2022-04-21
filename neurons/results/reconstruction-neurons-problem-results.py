#!/usr/bin/env python3
"""
GPLv3 2020 Miguel Aguilera

This code displays the results of the reconstruction Ising problem
computed from running "generate_data.py",  "inverse-Ising-problem.py"
and "reconstruction-Ising-problem.py"
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
import os

plt.rc('text', usetex=True)
font = {'size': 18, 'family':'serif', 'serif': ['latin modern roman']}
plt.rc('font', **font)
plt.rc('legend', **{'fontsize': 20})


size = 512
R = 1000000
R1 = 100000
J0 = 1.0
H0 = 0.5
Js = 0.1

T = 2**7
iu1 = np.triu_indices(size, 1)

offset = np.arange(3)
width = 0.20

B = 51
betas = np.linspace(0, 2, B)

N = len(betas)
M = 4
errorm = np.zeros((M, N))
errorC = np.zeros((M, N))
errorD = np.zeros((M, N))
errorH = np.zeros((M, N))
errorJ = np.zeros((M, N))

m_list = np.zeros(B)


CPexp = np.zeros(B)
C_list = np.zeros(B)


D_list = np.zeros(B)


dataset = 1


for ib in range(len(betas)):
    beta_ref = round(betas[ib], 4)
    print(ib, beta_ref)
    filename_r = 'dataset_1/reconstruction/transition_' + \
        str(int(round(beta_ref * 1000))) + '.npz'

    data_r = np.load(filename_r)

    m_final = data_r['m_final']
    C_final = data_r['C_final']
    D_final = data_r['D_final']


    m_list[ib] = np.mean(m_final)
    C_list[ib] = np.mean(C_final)
    D_list[ib] = np.mean(D_final)

    del data_r



labels = [
    r'Plefka[$t-1,t$]',
    r'Plefka[$t$]',
    r'Plefka[$t-1$]',
    r'Plefka2[$t$]',
    r'Original']
line = [(5, 4), (5, 4), '', '']
cmap = cm.get_cmap('plasma_r')
colors = []
for i in range(4):
    colors += [cmap((i) / 3)]
lws = [2, 2, 2, 3, 1.5]
pos_l = [-0.2, 1.0]

letters = ['A', 'B', 'C', 'D', 'E', 'F']

fig, ax = plt.subplots(1, 3, figsize=(16, 10 * 2 / 3), dpi=300)


nrow = 0
ncol = 0

ax[ncol].plot(betas, m_list, dashes=line[2],
                    color=colors[2], lw=lws[0])
ax[ncol].plot([1, 1], [0, 0.025], lw=0.5, color='k')
ax[ncol].axis([np.min(betas), np.max(betas), -1, 0])

ax[ncol].set_xlabel(r'$\beta / \beta_c$', fontsize=18)
ax[ncol].set_ylabel(
    r'$\mathrm{e}^{\langle \sigma_t\rangle}$', fontsize=18, rotation=0, labelpad=25)
ax[
    ncol].text(pos_l[0],
               pos_l[1],
               r'\textbf ' + letters[2],
               transform=ax[
                            ncol].transAxes,
               fontsize=20,
               va='top',
               ha='right')


nrow = 0
ncol = 1
ax[ncol].plot(betas, C_list, dashes=line[2], color=colors[2], lw=lws[2])
ax[ncol].plot(betas[np.argmax(C_list)], [0.0225],
                    '*', ms=10, color=colors[2])
ax[ncol].plot([1, 1], [0, 0.025], lw=0.5, color='k')
ax[ncol].axis([np.min(betas), np.max(betas), 0, 0.025])
ax[ncol].set_xlabel(r'$\beta / \beta_c$', fontsize=18)
ax[ncol].set_ylabel(
    r'$\langle C_{ik,t} \rangle$', fontsize=18, rotation=0, labelpad=25)
ax[
    ncol].text(pos_l[0],
               pos_l[1],
               r'\textbf ' + letters[0],
               transform=ax[
                            ncol].transAxes,
               fontsize=20,
               va='top',
               ha='right')

nrow = 0
ncol = 2
ax[ncol].plot(betas, D_list, dashes=line[2], color=colors[2], lw=lws[2])
ax[ncol].plot(betas[np.argmax(D_list)], [0.021],
                    '*', ms=10, color=colors[2])
ax[ncol].plot([1, 1], [0, 0.025], lw=0.5, color='k')
ax[ncol].axis([np.min(betas), np.max(betas), 0, 0.025])
ax[ncol].set_xlabel(r'$\beta / \beta_c$', fontsize=18)
ax[ncol].set_ylabel(
    r'$\langle D_{il,t} \rangle$', fontsize=18, rotation=0, labelpad=25)
ax[
    ncol].text(pos_l[0],
               pos_l[1],
               r'\textbf ' + letters[1],
               transform=ax[
                            ncol].transAxes,
               fontsize=20,
               va='top',
               ha='right')


folder = 'img/'
isExist = os.path.exists(folder)
if not isExist:
    # Create a new directory because it does not exist
    os.makedirs(folder)

plt.figlegend(
    loc='upper center',
    bbox_to_anchor=(
        0.5,
        1.),
    borderaxespad=0,
    ncol=5)
fig.tight_layout(h_pad=0.3, w_pad=0.7, rect=[0, 0, 1, 0.95])
plt.savefig(f'{folder}/dataset-{dataset}-neurons-reconstruted-transition.pdf', bbox_inches='tight')
