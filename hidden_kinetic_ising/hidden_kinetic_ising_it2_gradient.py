#!/usr/bin/env python3
"""

This code allows to run simulations of the kinetic Ising model,
with asymmetric weights and parallel updates.
"""

import numpy as np
from kinetic_ising import ising
import copy
import matplotlib.pyplot as plt
from utils import *
from hidden_kinetic_ising_it2 import HiddenIsing


def compute_all_gradients(hidden_ising, s, T_ori):

    T = T_ori

    num_samples = 10000
    parameters_list = np.linspace(-1.4, -1.1, num=num_samples)
    ell_list = np.zeros(num_samples)
    gradients_list = np.zeros(num_samples)

    for idx in range(0, num_samples):
        if idx%100==0:
            print(idx)

        hidden_ising.L[0,0] = parameters_list[idx]

        # Initialize the gradients to 0
        dLdH = np.zeros(hidden_ising.visible_size)
        dLdJ = np.zeros((hidden_ising.visible_size, hidden_ising.visible_size))
        dLdM = np.zeros((hidden_ising.visible_size, hidden_ising.b_size))
        dLdK = np.zeros((hidden_ising.b_size, hidden_ising.visible_size))
        dLdL = np.zeros((hidden_ising.b_size, hidden_ising.b_size))
        dLdb_0 = np.zeros(hidden_ising.b_size)

        # Likelihood accumulator
        log_ell = 0

        # State of b neurons at time [t-1]
        b_t1 = hidden_ising.b_0

        # In t==1 we need the derivative of b wrt K and L at t-1,
        # and that'd require s_{t-2} which does not exist at that time step
        db_t1_dK = np.zeros((hidden_ising.b_size, hidden_ising.b_size, hidden_ising.visible_size))
        db_t1_dL = np.zeros((hidden_ising.b_size, hidden_ising.b_size, hidden_ising.b_size))

        # db_dK = np.zeros((self.b_size, self.b_size, self.visible_size))
        db_dL = np.zeros((hidden_ising.b_size, hidden_ising.b_size, hidden_ising.b_size))

        # We start in index 1 because we do not have s_{t-1} for t=0
        for t in range(1, T):

            # Compute the derivative of the Likelihood wrt J
            h = hidden_ising.compute_h(s[t - 1], b_t1)
            tanh_h = np.tanh(h)
            # print(tanh_h)
            # print('h', np.dot(self.M, b_t1) + np.dot(self.J, s[t - 1]))
            # print('tanh_h', tanh_h)
            sub_s_tanhh = s[t] - tanh_h
            # print('sub_s_tanhh', sub_s_tanhh)

            # Compute the log Likelihood to check
            log_ell += np.dot(s[t], h) - np.sum(np.log(2 * np.cosh(h)))

            # Derivative of the Likelihood wrt H
            dLdH += sub_s_tanhh

            # Derivative of the Likelihood wrt J
            dLdJ += np.einsum('i,j->ij', sub_s_tanhh, s[t - 1])

            # Save computational load if the number of b neurons < 1
            if hidden_ising.b_size > 0:
                dLdM += np.einsum('i,j->ij', sub_s_tanhh, b_t1)

                if t == 1:
                    # Compute the gradient of the Likelihood wrt b(0) at t==1
                    dLdb_0 = np.dot(sub_s_tanhh, hidden_ising.M)
                if t == 2:
                    # Compute the gradient of the Likelihood wrt b(0) at t==2
                    b_t1_sq_rows = broadcast_rows((1 - b_t1 ** 2), hidden_ising.visible_size)
                    dLdb_0 += np.dot(sub_s_tanhh, np.einsum('ig,gz->iz', (hidden_ising.M * b_t1_sq_rows), hidden_ising.L))

                # Derivative of the Likelihood wrt K
                dLdK += np.einsum('i,inm->nm', sub_s_tanhh, np.einsum('ig,gnm->inm', hidden_ising.M, db_t1_dK))
                # Derivative of the Likelihood wrt L
                dLdL += np.einsum('i,inm->nm', sub_s_tanhh, np.einsum('ig,gnm->inm', hidden_ising.M, db_t1_dL))
                # for n in range(0, hidden_ising.b_size):
                #     for m in range(0, hidden_ising.b_size):
                #         for i in range(0, hidden_ising.visible_size):
                #             accum = 0
                #             for g in range(0, hidden_ising.b_size):
                #                 accum += hidden_ising.M[i,g] * db_t1_dL[g,n,m]
                #
                #             dLdL[n,m] += sub_s_tanhh[i] * accum

                # print(dLdL)

                # Compute the necessary information for the next step
                # At t==1, b(t-1)=0
                b = hidden_ising.compute_b(s[t - 1], b_t1)
                # print('b', b)

                # At t==1 db_t1_dK=0 and db_t1_dL=0
                # Derivative of b wrt K
                db_dK = np.einsum('gk,knm->gnm', hidden_ising.L, db_t1_dK)
                # Derivative of b wrt L
                db_dL = np.einsum('gk,knm->gnm', hidden_ising.L, db_t1_dL)
                for i in range(0, hidden_ising.b_size):
                    db_dK[i, i, :] += s[t - 1]
                    db_dK[i, :, :] *= (1 - b[i] ** 2)

                    db_dL[i, i, :] += b_t1
                    db_dL[i, :, :] *= (1 - b[i] ** 2)

                # db_dL = np.zeros((hidden_ising.b_size, hidden_ising.b_size, hidden_ising.b_size))
                # for n in range(0, hidden_ising.b_size):
                #     for m in range(0, hidden_ising.b_size):
                #         for g in range(0, hidden_ising.b_size):
                #             accum = 0
                #             for k in range(0, hidden_ising.b_size):
                #                 accum += hidden_ising.L[g,k] * db_t1_dL[k,n,m]
                #             db_dL[g,n,m] = accum
                #             if g == n:
                #                 db_dL[g, n, m] += b_t1[m]
                #
                #             db_dL[g, n, m] *= 1 - b[g]**2

                # print('dbdl', db_dL)


                # Save the variables for the next step
                b_t1 = copy.deepcopy(b)
                db_t1_dK = copy.deepcopy(db_dK)
                db_t1_dL = copy.deepcopy(db_dL)

        # Normalize the gradients temporally and by the number of spins in the sum of the Likelihood
        dLdH /= hidden_ising.visible_size * (T - 1)
        dLdJ /= hidden_ising.visible_size * (T - 1)
        dLdM /= hidden_ising.visible_size * (T - 1)
        dLdK /= hidden_ising.visible_size * (T - 1)
        dLdL /= hidden_ising.visible_size * (T - 1)
        dLdb_0 /= hidden_ising.visible_size * (T - 1)
        # log_ell /= (self.visible_size * (T - 1))

        ell_list[idx] = log_ell
        gradients_list[idx] = dLdL[0, 0]

    # np_gradients = np.gradient(ell_list, b0_list)
    np_gradients = np.gradient(ell_list, parameters_list) / (hidden_ising.visible_size * (T - 1))

    fig, ax = plt.subplots(2)

    #
    ax[0].plot(parameters_list, np_gradients, label='np_gradient')
    ax[0].plot(parameters_list, gradients_list, '--', label='computed gradient')
    ax[0].legend()
    ax[1].plot(parameters_list, ell_list, label='log(ell)')
    ax[1].set_xlabel('L[0,0]')
    ax[1].legend()
    plt.show()


if __name__ == "__main__":
    # You can set up a seed here for reproducibility
    # Seed to check wrong behavior: 6, 2425, 615
    # 3656, 0.6 1, 2, 3

    reproducible = True
    if reproducible:
        seed = 3412
    else:
        seed = np.random.randint(5000, size=1)

    rng = np.random.default_rng(seed)

    print('Seed', seed)

    original_netsize = 10
    vis_units = 7
    b_size = 3
    max_reps = 6500
    kinetic_ising = ising(netsize=original_netsize, rng=rng)
    kinetic_ising.random_fields()
    kinetic_ising.random_wiring()
    hidden_ising = HiddenIsing(kinetic_ising, visible_size=vis_units, rng=rng)
    hidden_ising.set_hidden_size(b_size=b_size)
    hidden_ising.random_wiring()

    T_ori = 500
    T_sim = 2000
    full_s, visible_s = hidden_ising.simulate_full(T_ori, burn_in=100)

    compute_all_gradients(hidden_ising, visible_s, T_ori)

    print('Seed', seed)
