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

#import context
from plefka import mf_ising
import numpy as np
import multiprocessing as mp


def nsf(num, n=4):
    """n-Significant Figures"""
    numstr = ("{0:.%ie}" % (n - 1)).format(num)
    return float(numstr)

def compute_mf_methods(beta, mode='f'):
    size = 512  # Network size
    R = 1000000  # Repetitions of the simulation
    H0 = 0.5  # Uniform distribution of fields parameter
    J0 = 1.0  # Average value of couplings
    Js = 0.1  # Standard deviation of couplings

    T = 2 ** 7  # Number of simulation time steps

    iu1 = np.triu_indices(size, 1)

    beta_ref = round(beta, 3)
    print(f'Beta {beta_ref}')

    # CREATE VARIABLES:

    mP_t1_t_mean = np.ones(T + 1)
    mP_CMS_mean = np.ones(T + 1)
    mP_t1_mean = np.ones(T + 1)

    CP_t1_t_mean = np.zeros(T + 1)
    CP_CMS_mean = np.zeros(T + 1)
    CP_t1_mean = np.zeros(T + 1)

    DP_t1_t_mean = np.zeros(T + 1)
    DP_CMS_mean = np.zeros(T + 1)
    DP_t1_mean = np.zeros(T + 1)

    # Load data

    filename = 'data/angel/inverse/inverse_100_R_' + str(R) + '.npz'
    #print(beta_ref, mode)

    data = np.load(filename)
    HP_t1_t = data['HP_t1_t'] * beta_ref
    JP_t1_t = data['JP_t1_t'] * beta_ref
    HP_t1 = data['HP_t1'] * beta_ref
    JP_t1 = data['JP_t1'] * beta_ref
    HP_CMS = data['HP2_t'] * beta_ref
    JP_CMS = data['JP2_t'] * beta_ref

    J = data['J'] * beta_ref
    H = data['H'] * beta_ref
    del data

    filename1 = 'data/data-H0-' + str(H0) + '-J0-' + str(J0) + '-Js-' + str(
        Js) + '-N-' + str(size) + '-R-' + str(R) + '-beta-1.0.npz'
    data1 = np.load(filename1)
    s0 = data1['s0']
    del data1

    # Run _CMS
    I = mf_ising(size)
    if mode == 'f':
        I.H = H.copy()
        I.J = J.copy()
    elif mode == 'r':
        I.H = HP_CMS.copy()
        I.J = JP_CMS.copy()
    I.initialize_state(s0)
    for t in range(T):
        # print(beta_ref, mode, 'P_CMS', str(t) + '/' + str(T),
        #       nsf(np.mean(I.m)), nsf(np.mean(I.C[iu1])), nsf(np.mean(I.D)))
        if t == 0:
            I.update_P_t1_o1()
        else:
            I.update_P_CMS()

        CP_CMS_mean[t + 1] = np.mean(I.C)
        mP_CMS_mean[t + 1] = np.mean(I.m)
        DP_CMS_mean[t + 1] = np.mean(I.D)

    mP_CMS_final = I.m
    CP_CMS_final = I.C
    DP_CMS_final = I.D

    # Run Plefka[t-1,t], order 2 TAP
    I = mf_ising(size)
    if mode == 'f':
        I.H = H.copy()
        I.J = J.copy()
    elif mode == 'r':
        I.H = HP_t1_t.copy()
        I.J = JP_t1_t.copy()
    I.initialize_state(s0)
    for t in range(T):
        # print(beta_ref, mode, 'P_t1_t_o2', str(t) + '/' + str(T),
        #       nsf(np.mean(I.m)), nsf(np.mean(I.C[iu1])), nsf(np.mean(I.D)))
        I.update_P_t1_t_o2()

        CP_t1_t_mean[t + 1] = np.mean(I.C)
        mP_t1_t_mean[t + 1] = np.mean(I.m)
        DP_t1_t_mean[t + 1] = np.mean(I.D)

    mP_t1_t_final = I.m
    CP_t1_t_final = I.C
    DP_t1_t_final = I.D

    # Run Plefka[t-1], order 1 MS
    I = mf_ising(size)
    if mode == 'f':
        I.H = H.copy()
        I.J = J.copy()
    elif mode == 'r':
        I.H = HP_t1.copy()
        I.J = JP_t1.copy()
    I.initialize_state(s0)
    for t in range(T):
        # print(beta_ref, mode, 'P_t1_o1', str(t) + '/' + str(T),
        #       nsf(np.mean(I.m)), nsf(np.mean(I.C[iu1])), nsf(np.mean(I.D)))
        I.update_P_t1_o1()

        CP_t1_mean[t + 1] = np.mean(I.C)
        mP_t1_mean[t + 1] = np.mean(I.m)
        DP_t1_mean[t + 1] = np.mean(I.D)

    mP_t1_final = I.m
    CP_t1_final = I.C
    DP_t1_final = I.D

    # Save results to file

    filename = 'data/reconstruction/transition_' + mode + '_' + \
               str(int(round(beta_ref * 1000))) + '_R_' + str(R) + '.npz'
    np.savez_compressed(filename,
                        mP_t1_t=mP_t1_t_final,
                        mP_t1=mP_t1_final,
                        mP2_t=mP_CMS_final,
                        CP_t1_t=CP_t1_t_final,
                        CP_t1=CP_t1_final,
                        CP2_t=CP_CMS_final,
                        DP_t1_t=DP_t1_t_final,
                        DP_t1=DP_t1_final,
                        DP2_t=DP_CMS_final,
                        mP_t1_t_mean=mP_t1_t_mean,
                        mP_CMS_mean=mP_CMS_mean,
                        mP_t1_mean=mP_t1_mean,
                        CP_t1_t_mean=CP_t1_t_mean,
                        CP_CMS_mean=CP_CMS_mean,
                        CP_t1_mean=CP_t1_mean,
                        DP_t1_t_mean=DP_t1_t_mean,
                        DP_CMS_mean=DP_CMS_mean,
                        DP_t1_mean=DP_t1_mean,
                        )


if __name__ == '__main__':
    B = 201  # Number of values of beta

    betas = 1 + np.linspace(-1, 1, B) * 0.3

    betas = [beta for beta in betas if beta > 0.9]

    # As we need more data than what is generated in the forward problem,
    # we compute the forward problem twice, for mode='f' and mode='r' for
    # generating data for the forward and reconstruction problems
    # modes = ['f', 'r']        # Forward and reconstruction modes
    # modes = ['r']
    pool = mp.Pool(4)
    pool.map(compute_mf_methods, betas)

    pool.close()





