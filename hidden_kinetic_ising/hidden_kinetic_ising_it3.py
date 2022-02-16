#!/usr/bin/env python3
"""

This code allows to run simulations of the kinetic Ising model,
with asymmetric weights and parallel updates.
"""

from kinetic_ising import ising
from utils.common_functions import averaged_MSE

# !/usr/bin/env python3
"""

This code allows to run simulations of the kinetic Ising model,
with asymmetric weights and parallel updates.
"""
import numpy as np
import copy


class HiddenIsing:  # Asymmetric Ising model with hidden activity simulation class

    def __init__(self, visible_size, rng=None):  # Create ising model
        """
        Initializes the class for simulation

        :param original_ising: ising model you want to learn from
        :param visible_size: number of visible unit
        :param b_size: number of b type "hidden" neurons
        :param rng: random number generator. If not set, one is created.
        """

        self.Beta = 1  # Inverse temperature

        if rng:
            self.rng = rng
        else:
            self.rng = np.random.default_rng()

        self.visible_size = visible_size  # Network size

        self.H = np.zeros(self.visible_size)  # External fields
        self.J = np.zeros((self.visible_size, self.visible_size))  # Spin-to-Spin couplings
        self.K = np.zeros((self.visible_size, self.visible_size))  # Hidden-to-Neuron couplings
        self.L = np.zeros((self.visible_size, self.visible_size))  # Hidden-to-Hidden couplings
        self.b_0 = np.zeros(self.visible_size)
        self.h_0 = np.zeros(self.visible_size)

    def random_wiring(self):  # Set random values for J
        self.H = self.rng.random(self.visible_size) * 2 - 1
        self.J = self.rng.random((self.visible_size, self.visible_size)) / self.visible_size
        self.K = self.rng.random((self.visible_size, self.visible_size)) / self.visible_size
        self.L = self.rng.random((self.visible_size, self.visible_size)) / self.visible_size
        self.b_0 = self.rng.random(self.visible_size) * 2 - 1
        self.h_0 = self.rng.random(self.visible_size) * 2 - 1

        print('H', self.H)
        print('J', self.J)
        print('K', self.K)
        print('L', self.L)
        print('b_0', self.b_0)
        print()

    def gradient_descent(self, dLdH, dLdJ, dLdK, dLdL, eta, mode='regular'):

        if mode == 'regular':
            error = self.regular_gradient_descent(dLdH, dLdJ, dLdK, dLdL, eta)
        elif mode == 'coordinated':
            error = self.coordinated_gradient_descent(dLdH, dLdJ, dLdK, dLdL, eta)

        return error

    def regular_gradient_descent(self, dLdH, dLdJ, dLdK, dLdL, eta):

        error = max(np.max(np.abs(dLdH)), np.max(np.abs(dLdJ)), np.max(np.abs(dLdK)),
                    np.max(np.abs(dLdL)))
        self.H = self.H + eta * dLdH
        self.J = self.J + eta * dLdJ
        self.K = self.K + eta * dLdK
        self.L = self.L + eta * dLdL
        # self.b_0 = self.b_0 + eta * dLdb_0
        # self.h_0 = self.h_0 + eta * dLdh_0

        return error

    def coordinated_gradient_descent(self, dLdH, dLdJ, dLdK, dLdL, eta):

        max_H_idx = np.argmax(np.abs(dLdH))
        max_J_idx = np.argmax(np.abs(dLdJ))
        max_J_idx = np.unravel_index(max_J_idx, dLdJ.shape)
        max_K_idx = np.argmax(np.abs(dLdK))
        max_K_idx = np.unravel_index(max_K_idx, dLdK.shape)
        max_L_idx = np.argmax(np.abs(dLdL))
        max_L_idx = np.unravel_index(max_L_idx, dLdL.shape)
        # max_b0_idx = np.argmax(np.abs(dLdb_0))
        # max_b0_idx = np.unravel_index(max_b0_idx, dLdb_0.shape)
        # max_h0_idx = np.argmax(np.abs(dLdh_0))
        # max_h0_idx = np.unravel_index(max_h0_idx, dLdh_0.shape)

        max_H = dLdH[max_H_idx]
        max_J = dLdJ[max_J_idx]
        max_K = dLdK[max_K_idx]
        max_L = dLdL[max_L_idx]
        # max_b0 = dLdb_0[max_b0_idx]
        # max_h0 = dLdh_0[max_h0_idx]

        max_max = np.argmax(np.abs([max_H, max_J, max_K, max_L]))
        # print(max_max)
        if max_max == 0:
            self.H[max_H_idx] = self.H[max_H_idx] + eta * max_H
            error = np.abs(max_H)

        elif max_max == 1:
            self.J[max_J_idx] = self.J[max_J_idx] + eta * max_J
            error = np.abs(max_J)

        elif max_max == 2:
            self.K[max_K_idx] = self.K[max_K_idx] + eta * max_K
            error = np.abs(max_K)

        elif max_max == 3:
            self.L[max_L_idx] = self.L[max_L_idx] + eta * max_L
            error = np.abs(max_L)

        # elif max_max == 4:
        #     self.b_0[max_b0_idx] = self.b_0[max_b0_idx] + eta * max_b0
        #     error = np.abs(max_b0)
        #
        # elif max_max == 5:
        #     self.h_0[max_h0_idx] = self.h_0[max_h0_idx] + eta * max_h0
        #     error = np.abs(max_h0)

        else:
            print('Error, no gradient has been updated')

        return error

    def compute_b(self, s_t1, b_t1, verbose=False):
        """

        :param s_t1: State of the spins at time t-1
        :param b_t1: State of the hidden neurons at time t-1
        :return: state of the hidden neurons at time t
        """
        if verbose:
            print('dot K s', np.dot(self.K, s_t1))
            print('np.dot(self.L, np.tanh(b_t1))', np.dot(self.L, np.tanh(b_t1)))
        return np.dot(self.K, s_t1) + np.dot(self.L, np.tanh(b_t1))

    def compute_h(self, s_t1, b):
        """

        :param s_t1: State of the spins at time t-1
        :param b_t1: State of the hidden neurons at time t-1
        :return: effective field at time t
        """

        return self.H + b + np.dot(self.J, s_t1)

    def fit(self, s, eta, max_reps, T_ori, T_sim, original_moments, num_simulations=5, burn_in=0, gradient_mode='regular'):

        """

        :param s: evolution of the system trough T steps
        :param eta:
        :param max_reps:
        :param T_ori:
        :param T_sim:
        :param gradient_mode: type of gradient descent to perform. Can be either 'regular' or 'coordinated'
        :return:
        """


        # Initialize variables for learning
        rep = 0
        error_lim = 0.0000001
        error = np.inf
        # Learning loop
        T = T_ori

        ell_list = np.zeros(max_reps)
        error_list = np.zeros(max_reps)
        MSE_m_list = []
        MSE_C_list = []
        MSE_D_list = []
        error_iter_list = []

        plot_interval = 250

        old_error = np.inf

        log_ell_t1 = -np.inf

        while error > error_lim and rep < max_reps:

            # if rep % plot_interval == 0:
            #     print('Iter', rep)

            # Initialize the gradients to 0
            dLdH = np.zeros(self.visible_size)
            dLdJ = np.zeros((self.visible_size, self.visible_size))
            dLdK = np.zeros((self.visible_size, self.visible_size))
            dLdL = np.zeros((self.visible_size, self.visible_size))

            # Likelihood accumulator
            log_ell = 0

            # State of b at time [t-1]
            b_t1 = self.b_0

            # In t==1 we need the derivative of b wrt K and L at t-1,
            # and that'd require s_{t-2} which does not exist at that time step
            db_t1_dK = np.zeros((self.visible_size, self.visible_size, self.visible_size))
            db_t1_dL = np.zeros((self.visible_size, self.visible_size, self.visible_size))

            s_t1 = np.squeeze(np.asarray(s[0]))

            # h = self.h_0
            # sub_s_tanhh = s_t1 - np.tanh(h)
            # dLdb_0 = sub_s_tanhh
            # dLdh_0 = sub_s_tanhh
            # log_ell += np.dot(s_t1, h) - np.sum(np.log(2 * np.cosh(h)))

            # We start in index 1 because we do not have s_{t-1} for t=0
            for t in range(1, T):

                s_t = np.squeeze(np.asarray(s[t]))

                # Compute the effective field of every neuron
                b = self.compute_b(s_t1, b_t1)
                h = self.compute_h(s_t1, b)
                tanh_h = np.tanh(h)
                sub_s_tanhh = s_t - tanh_h

                # Compute the log Likelihood to check
                log_ell += np.dot(s_t, h) - np.sum(np.log(2 * np.cosh(h)))

                # Derivative of the Likelihood wrt H
                dLdH += sub_s_tanhh

                # Derivative of the Likelihood wrt J
                dLdJ += np.einsum('i,j->ij', sub_s_tanhh, s_t1)

                # if t == 1:
                #     # Compute the gradient of the Likelihood wrt b(0) at t==1
                #      b_tanh_sq_rows = broadcast_rows((1 - np.tanh(b_t1) ** 2), self.visible_size)
                #      dLdb_0 += np.dot(sub_s_tanhh, b_tanh_sq_rows*self.L)

                # At t==1 b_t1_dK=0 and b_t1_dL=0
                # Derivative of b wrt K
                db_dK = np.einsum('ig,gnm->inm', self.L * (1 - np.tanh(b_t1) ** 2), db_t1_dK)
                # Derivative of b wrt L
                db_dL = np.einsum('ig,gnm->inm', self.L * (1 - np.tanh(b_t1) ** 2), db_t1_dL)
                for i in range(0, self.visible_size):
                    db_dK[i, i, :] += s_t1
                    db_dL[i, i, :] += np.tanh(b_t1)

                # Compute the Jacobians
                # Derivative of the Likelihood wrt K
                dLdK += np.einsum('i,inm->nm', sub_s_tanhh, db_dK)
                # Derivative of the Likelihood wrt L
                dLdL += np.einsum('i,inm->nm', sub_s_tanhh, db_dL)

                # Save the variables for the next step
                s_t1 = copy.deepcopy(s_t)
                b_t1 = copy.deepcopy(b)
                db_t1_dK = copy.deepcopy(db_dK)
                db_t1_dL = copy.deepcopy(db_dL)

            # Normalize the gradients temporally and by the number of spins in the sum of the Likelihood
            dLdH /= self.visible_size * (T - 1)
            dLdJ /= self.visible_size * (T - 1)
            dLdK /= self.visible_size * (T - 1)
            dLdL /= self.visible_size * (T - 1)
            # dLdb_0 /= self.visible_size * (T - 1)
            # dLdh_0 /= self.visible_size * (T - 1)
            log_ell /= self.visible_size * (T - 1)

            # error = self.gradient_descent(dLdH, dLdJ, dLdK, dLdL, dLdb_0, dLdh_0, eta, mode=gradient_mode)
            error = self.gradient_descent(dLdH, dLdJ, dLdK, dLdL, eta, mode=gradient_mode)

            if (rep % plot_interval == 0) or rep == (max_reps - 1):
                # Compute the averaged MSE for m, C and D
                MSE_m, MSE_C, MSE_D = averaged_MSE(self, original_moments, T_sim, num_simulations, burn_in=burn_in)

                MSE_m_list.append(MSE_m)
                MSE_C_list.append(MSE_C)
                MSE_D_list.append(MSE_D)
                error_iter_list.append(rep)

            error_list[rep] = error

            if log_ell_t1 > log_ell:
                print('#################################### WRONG | LIKELIHOOD DECREASING')

            # print('Comparison', '(log_ell-log_ell_t1)/eta', (log_ell-log_ell_t1)/eta, 'dLdb_0²', old_error_b0**2)

            ell_list[rep] = log_ell
            log_ell_t1 = log_ell

            rep = rep + 1

            # print()

        print('dLdH', dLdH)
        print('dLdL', dLdL)

        print('dLdJ', dLdJ)
        print('dLdK', dLdK)
        # print('dLdb0', dLdb_0)

        return ell_list[:rep], error_list[:rep], MSE_m_list, MSE_C_list, MSE_D_list, error_iter_list

    # Update the state of the network using Little parallel update rule
    def hidden_parallel_update(self, sim_s_t1, b):
        h = self.compute_h(sim_s_t1, b)
        r = self.rng.random(self.visible_size)
        sim_s_t1 = -1 + 2 * (2 * self.Beta * h > - np.log(1 / r - 1)).astype(int)

        return sim_s_t1

    def simulate_visible(self, T, burn_in=0):
        # Initialize all visible neurons to -1
        sim_s = np.ones(self.visible_size) * -1
        visible_s_matrix = np.zeros((T, self.visible_size))
        b_t1 = self.b_0

        for t in range(0, T + burn_in):

            b = self.compute_b(sim_s, b_t1)
            sim_s = self.hidden_parallel_update(sim_s, b)
            if t >= burn_in:
                visible_s_matrix[t - burn_in] = sim_s

            b_t1 = copy.deepcopy(b)

        return visible_s_matrix

