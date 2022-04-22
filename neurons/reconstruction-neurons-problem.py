#!/usr/bin/env python3
"""
GPLv3 2020 Miguel Aguilera

This code is similar to "forward-Ising-problem.py", but it is used
for solving the reconstruction problem, so it saves just the values of
correlations at the end of the simulation.
It computes the solution of the forward Ising problem with different
mean-field approximation methods using either the original network
mode='f', or the inferred network mode='r' from solving the inverse
Ising problem.
The results can be displayed running "reconstruction-Ising-problem-results.py"
"""

import numpy as np
import os
from simulate import update_moments_gamma

def nsf(num, n=4):
    """n-Significant Figures"""
    numstr = ("{0:.%ie}" % (n - 1)).format(num)
    return float(numstr)


if __name__ == "__main__":
    B = 51                    # Number of values of beta
    T = 128                  # Number of simulation time steps

    betas = np.linspace(0, 2, B)

    dataset = 1
    filename = f'results/dataset_{dataset}/stats/stats' + str(dataset) + '.npz'
    data = np.load(filename)
    N = data['N']
    Cexp = data['C']
    Dexp = data['D']
    mexp = data['m']

    filename = f'results/dataset_{dataset}/networks/net_' + str(dataset) + '.npz'
    data = np.load(filename)
    N = data['N']
    H = data['H']
    J = data['J']
    gamma = data['gamma']

    # As we need more data than what is generated in the forward problem,
    # we compute the forward problem twice, for mode='f' and mode='r' for
    # generating data for the forward and reconstruction problems

    for ib in range(len(betas)):
        beta_ref = round(betas[ib], 3)

        J_beta = J * beta_ref
        H_beta = H * beta_ref

        m = mexp.copy()
        C = Cexp.copy() * 0
        D = Dexp.copy() * 0

        for t in range(T):
            m_p = m.copy()
            C_p = C.copy()
            D_p = D.copy()
            m, C, D = update_moments_gamma(H_beta, J_beta, N, m_p, C_p, gamma)

            if t % 10 == 0 or t == T-1:
                print(beta_ref, 'H', np.mean(H_beta), 'J', np.mean(J_beta), 'P_t1_t_o2', str(t) + '/' + str(T))

        m_final = m
        C_final = C
        D_final = D

        folder = f'results/dataset_{dataset}/reconstruction/'
        isExist = os.path.exists(folder)
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(folder)

        # Save results to file
        filename = folder + 'transition_' + \
            str(int(round(beta_ref * 1000))) + '.npz'
        np.savez_compressed(filename,
                            m_final=m_final,
                            C_final=C_final,
                            D_final=D_final)
