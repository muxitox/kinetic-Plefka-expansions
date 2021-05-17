#!/usr/bin/env python3
"""
GPLv3 2020 Miguel Aguilera

This code computes the solution of the forward Ising problem with different
mean-field approximation methods and compares it with the averages and
correlations obtained from simulation from "generate_data.py".
The results can be displayed running "forward-Ising-problem-results.py"
"""
from plefka import mf_ising
import numpy as np
import time


def nsf(num, n=4):
    """n-Significant Figures"""
    numstr = ("{0:.%ie}" % (n - 1)).format(num)
    return float(numstr)


size = 512                 # Network size
R = 1000000                # Repetitions of the simulation
H0 = 0.5                   # Uniform distribution of fields parameter
J0 = 1.0                   # Average value of couplings
Js = 0.1                   # Standard deviation of couplings

B = 21                    # Number of values of beta
T = 2**7                  # Number of simulation time steps

betas = 1 + np.linspace(-1, 1, B) * 0.3
for ib in range(B):
    print(f'Beta {betas[ib]}')
    beta_ref = round(betas[ib], 3)

    EmP_t1_t = np.zeros(T + 1)
    EmP_CMS = np.zeros(T + 1)
    EmP_t1 = np.zeros(T + 1)

    ECP_t1_t = np.zeros(T + 1)
    ECP_CMS = np.zeros(T + 1)
    ECP_t1 = np.zeros(T + 1)

    EDP_t1_t = np.zeros(T + 1)
    EDP_t = np.zeros(T + 1)
    EDP_CMS = np.zeros(T + 1)
    EDP_t1 = np.zeros(T + 1)

    mP_t1_t_mean = np.ones(T + 1)
    mP_CMS_mean = np.ones(T + 1)
    mPexp_mean = np.ones(T + 1)
    mP_t1_mean = np.ones(T + 1)
    mP_t1_t_final = np.zeros(size)
    mP_CMS_final = np.zeros(size)
    mP_t1_final = np.zeros(size)
    mPexp_final = np.zeros(size)

    CP_t1_t_mean = np.zeros(T + 1)
    CP_CMS_mean = np.zeros(T + 1)
    CPexp_mean = np.zeros(T + 1)
    CP_t1_mean = np.zeros(T + 1)
    CP_t1_t_final = np.zeros((size, size))
    CP_CMS_final = np.zeros((size, size))
    CP_t1_final = np.zeros((size, size))
    CPexp_final = np.zeros((size, size))

    DP_t1_t_mean = np.zeros(T + 1)
    DP_CMS_mean = np.zeros(T + 1)
    DPexp_mean = np.zeros(T + 1)
    DP_t1_mean = np.zeros(T + 1)
    DP_t1_t_final = np.zeros((size, size))
    DP_CMS_final = np.zeros((size, size))
    DP_t1_final = np.zeros((size, size))
    DPexp_final = np.zeros((size, size))

    # Load data
    filename = 'data/data-H0-' + str(H0) + '-J0-' + str(J0) + '-Js-' + str(
        Js) + '-N-' + str(size) + '-R-' + str(R) + '-beta-' + str(beta_ref) + '.npz'
    print(filename)
    data = np.load(filename)
    H = data['H']
    J = data['J']
    s0 = data['s0']
    m_exp = data['m']
    C_exp = data['C']
    D_exp = data['D']
    del data

    # Load statistical moments from data
    mPexp_mean[0] = 1
    for t in range(T):
        mPexp_mean[t + 1] = np.mean(m_exp[:, t])
        CPexp_mean[t + 1] = np.mean(C_exp[:, :, t])
        DPexp_mean[t + 1] = np.mean(D_exp[:, :, t])
        print('Exp',
              str(t) + '/' + str(T),
              mPexp_mean[t + 1],
              CPexp_mean[t + 1],
              DPexp_mean[t + 1])
    mPexp_final = m_exp[:, T - 1]
    CPexp_final = C_exp[:, :, T - 1]
    DPexp_final = D_exp[:, :, T - 1]

    # Initialize kinetic Ising model
    I = mf_ising(size)
    I.H = H.copy()
    I.J = J.copy()

    # Run Plefka2[t-1], order 2 CMS
    time_start = time.perf_counter()
    I.initialize_state(s0)
    for t in range(T):
        if t == 0:
            I.update_P_t1_o1()
        else:
            I.update_P_CMS()
        CP_CMS_mean[t + 1] = np.mean(I.C)
        mP_CMS_mean[t + 1] = np.mean(I.m)
        DP_CMS_mean[t + 1] = np.mean(I.D)
        EmP_CMS[t + 1] = np.mean((I.m - m_exp[:, t]) ** 2)
        ECP_CMS[t + 1] = np.mean((I.C - C_exp[:, :, t]) ** 2)
        EDP_CMS[t + 1] = np.mean((I.D - D_exp[:, :, t]) ** 2)
        print('beta',
              beta_ref,
              'P_CMS',
              str(t) + '/' + str(T),
              nsf(mP_CMS_mean[t + 1]),
              nsf(CP_CMS_mean[t + 1]),
              nsf(DP_CMS_mean[t + 1]),
              nsf(EmP_CMS[t + 1]),
              nsf(ECP_CMS[t + 1]),
              nsf(EDP_CMS[t + 1]))
    mP_CMS_final = I.m
    CP_CMS_final = I.C
    DP_CMS_final = I.D
    time_P_CMS = time.perf_counter() - time_start

    # Run Plefka[t-1,t], order 2 TAP
    time_start = time.perf_counter()
    I.initialize_state(s0)
    for t in range(T):
        I.update_P_t1_t_o2()
        CP_t1_t_mean[t + 1] = np.mean(I.C)
        mP_t1_t_mean[t + 1] = np.mean(I.m)
        DP_t1_t_mean[t + 1] = np.mean(I.D)
        EmP_t1_t[t + 1] = np.mean((I.m - m_exp[:, t])**2)
        ECP_t1_t[t + 1] = np.mean((I.C - C_exp[:, :, t])**2)
        EDP_t1_t[t + 1] = np.mean((I.D - D_exp[:, :, t])**2)
        print('beta',
              beta_ref,
              'P_t1_t_o2',
              str(t) + '/' + str(T),
              nsf(mP_t1_t_mean[t + 1]),
              nsf(CP_t1_t_mean[t + 1]),
              nsf(DP_t1_t_mean[t + 1]),
              nsf(EmP_t1_t[t + 1]),
              nsf(ECP_t1_t[t + 1]),
              nsf(EDP_t1_t[t + 1]))
    mP_t1_t_final = I.m
    CP_t1_t_final = I.C
    DP_t1_t_final = I.D
    time_P_t1_t = time.perf_counter() - time_start

    # Run Plefka[t-1], order 1
    time_start = time.perf_counter()
    I.initialize_state(s0)
    for t in range(T):
        I.update_P_t1_o1()
        CP_t1_mean[t + 1] = np.mean(I.C)
        mP_t1_mean[t + 1] = np.mean(I.m)
        DP_t1_mean[t + 1] = np.mean(I.D)
        EmP_t1[t + 1] = np.mean((I.m - m_exp[:, t])**2)
        ECP_t1[t + 1] = np.mean((I.C - C_exp[:, :, t])**2)
        EDP_t1[t + 1] = np.mean((I.D - D_exp[:, :, t])**2)
        print('beta',
              beta_ref,
              'P_t1_o1',
              str(t) + '/' + str(T),
              nsf(mP_t1_mean[t + 1]),
              nsf(CP_t1_mean[t + 1]),
              nsf(DP_t1_mean[t + 1]),
              nsf(EmP_t1[t + 1]),
              nsf(ECP_t1[t + 1]),
              nsf(EDP_t1[t + 1]))
    mP_t1_final = I.m
    CP_t1_final = I.C
    DP_t1_final = I.D
    time_P_t1 = time.perf_counter() - time_start

    # Save results to file

    filename = 'data/forward/forward_' + \
        str(int(beta_ref * 100)) + '_R_' + str(R) + '.npz'
    np.savez_compressed(filename,
                        m_exp=m_exp[:, T - 1],
                        C_exp=C_exp[:, :, T - 1],
                        D_exp=D_exp[:, :, T - 1],
                        mP_t1_t_mean=mP_t1_t_mean,
                        mP_t1_mean=mP_t1_mean,
                        mP_CMS_mean=mP_CMS_mean,
                        CP_t1_t_mean=CP_t1_t_mean,
                        CP_t1_mean=CP_t1_mean,
                        CP_CMS_mean=CP_CMS_mean,
                        DP_t1_t_mean=DP_t1_t_mean,
                        DP_t1_mean=DP_t1_mean,
                        DP_CMS_mean=DP_CMS_mean,
                        mPexp_mean=mPexp_mean,
                        CPexp_mean=CPexp_mean,
                        DPexp_mean=DPexp_mean,
                        mP_t1_t=mP_t1_t_final,
                        mP_t1=mP_t1_final,
                        mP_CMS=mP_CMS_final,
                        CP_t1_t=CP_t1_t_final,
                        CP_t1=CP_t1_final,
                        CP_CMS=CP_CMS_final,
                        DP_t1_t=DP_t1_t_final,
                        DP_t1=DP_t1_final,
                        DP_CMS=DP_CMS_final,
                        EmP_t1_t=EmP_t1_t,
                        EmP_t1=EmP_t1,
                        Em_CMS=EmP_CMS,
                        ECP_t1_t=ECP_t1_t,
                        ECP_t1=ECP_t1,
                        ECP_CMS=ECP_CMS,
                        EDP_t1_t=EDP_t1_t,
                        EDP_t=EDP_t,
                        EDP_t1=EDP_t1,
                        EDP_CMS=EDP_CMS,
                        time_P_t1_t=time_P_t1_t,
                        time_P_t1=time_P_t1,
                        time_P_CMS=time_P_CMS)
