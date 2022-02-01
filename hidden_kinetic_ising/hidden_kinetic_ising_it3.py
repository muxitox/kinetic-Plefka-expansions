#!/usr/bin/env python3
"""

This code allows to run simulations of the kinetic Ising model,
with asymmetric weights and parallel updates.
"""

import numpy as np
from kinetic_ising import ising
import random
import copy


#!/usr/bin/env python3
"""

This code allows to run simulations of the kinetic Ising model,
with asymmetric weights and parallel updates.
"""
import numpy as np
import copy
from utils import *


class HiddenIsing:  # Asymmetric Ising model with hidden activity simulation class

    def __init__(self, original_ising, visible_size, rng=None):  # Create ising model
        """
        Initializes the class for simulation

        :param original_ising: ising model you want to learn from
        :param visible_size: number of visible unit
        :param b_size: number of b type "hidden" neurons
        :param rng: random number generator. If not set, one is created.
        """

        self.ising = original_ising
        self.size = self.ising.size

        self.Beta = 1  # Inverse temperature

        if rng:
            self.rng = rng
        else:
            self.rng = np.random.default_rng()

        self.visible_size = visible_size  # Network size

        self.visible_idx = self.rng.choice(range(0, self.ising.size), self.visible_size)

        self.H = np.zeros(self.visible_size)  # External fields
        self.J = np.zeros((self.visible_size, self.visible_size))  # Spin-to-Spin couplings
        self.K = np.zeros((self.visible_size, self.visible_size))  # Hidden-to-Neuron couplings
        self.L = np.zeros((self.visible_size, self.visible_size))  # Hidden-to-Hidden couplings
        self.b_0 = np.zeros(self.visible_size)



    def random_wiring(self):  # Set random values for J
        self.H = self.rng.random(self.visible_size) * 2 - 1
        self.J = self.rng.random((self.visible_size, self.visible_size)) / self.visible_size
        self.K = self.rng.random((self.visible_size, self.visible_size)) / self.visible_size
        self.L = self.rng.random((self.visible_size, self.visible_size)) / self.visible_size
        self.b_0 = self.rng.random(self.visible_size) * 2 - 1

        print('H', self.H)
        print('J', self.J)
        print('K', self.K)
        print('L', self.L)
        print('b_0', self.b_0)
        print()

    def gradient_descent(self, dLdH, dLdJ, dLdK, dLdL, dLdb_0, eta, mode='regular'):

        if mode == 'regular':
            error = self.regular_gradient_descent(dLdH, dLdJ, dLdK, dLdL, dLdb_0, eta)
        elif mode == 'coordinated':
            error = self.coordinated_gradient_descent(dLdH, dLdJ, dLdK, dLdL, dLdb_0, eta)

        return error

    def regular_gradient_descent(self, dLdH, dLdJ, dLdK, dLdL, dLdb_0, eta):

        error = max(np.max(np.abs(dLdH)), np.max(np.abs(dLdJ)), np.max(np.abs(dLdM)), np.max(np.abs(dLdK)),
                    np.max(np.abs(dLdL)), np.max(dLdb_0))
        self.H = self.H + eta * dLdH
        self.J = self.J + eta * dLdJ
        self.K = self.K + eta * dLdK
        self.L = self.L + eta * dLdL
        self.b_0 = self.b_0 + eta * dLdb_0


        return error

    def coordinated_gradient_descent(self, dLdH, dLdJ, dLdK, dLdL, dLdb_0, eta):

        max_H_idx = np.argmax(np.abs(dLdH))
        max_J_idx = np.argmax(np.abs(dLdJ))
        max_J_idx = np.unravel_index(max_J_idx, dLdJ.shape)
        max_K_idx = np.argmax(np.abs(dLdK))
        max_K_idx = np.unravel_index(max_K_idx, dLdK.shape)
        max_L_idx = np.argmax(np.abs(dLdL))
        max_L_idx = np.unravel_index(max_L_idx, dLdL.shape)
        max_b0_idx = np.argmax(np.abs(dLdb_0))
        max_b0_idx = np.unravel_index(max_b0_idx, dLdb_0.shape)

        max_H = dLdH[max_H_idx]
        max_J = dLdJ[max_J_idx]
        max_K = dLdK[max_K_idx]
        max_L = dLdL[max_L_idx]
        max_b0 = dLdb_0[max_b0_idx]

        max_max = np.argmax(np.abs([max_H, max_J, max_K, max_L, max_b0]))
        if max_max == 0:
            self.H[max_H_idx] = self.H[max_H_idx] + eta * max_H
            error = np.abs(max_H)

        if max_max == 1:
            self.J[max_J_idx] = self.J[max_J_idx] + eta * max_J
            error = np.abs(max_J)

        elif max_max == 2:
            self.K[max_K_idx] = self.K[max_K_idx] + eta * max_K
            error = np.abs(max_K)

        elif max_max == 3:
            self.L[max_L_idx] = self.L[max_L_idx] + eta * max_L
            error = np.abs(max_L)

        elif max_max == 4:
            self.b_0[max_b0_idx] = self.b_0[max_b0_idx] + eta * max_b0
            error = np.abs(max_b0)

        return error

    def compute_b(self, s_t1, b_t1):
        """

        :param s_t1: State of the spins at time t-1
        :param b_t1: State of the hidden neurons at time t-1
        :return: state of the hidden neurons at time t
        """
        return np.dot(self.K, s_t1) + np.dot(self.L, np.tanh(b_t1))

    def compute_h(self, s_t1, b):
        """

        :param s_t1: State of the spins at time t-1
        :param b_t1: State of the hidden neurons at time t-1
        :return: effective field at time t
        """

        return self.H + b + np.dot(self.J, s_t1)

    def fit(self, s, eta, max_reps, T_ori, T_sim, original_moments, gradient_mode='regular'):

        """

        :param s: evolution of the system trough T steps
        :param eta:
        :param max_reps:
        :param T_ori:
        :param T_sim:
        :param gradient_mode: type of gradient descent to perform. Can be either 'regular' or 'coordinated'
        :return:
        """

        m, C, D = original_moments

        # Initialize variables for learning
        rep = 0
        error_lim = 0.000005
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
            dLdb_0 = np.zeros(self.visible_size)
            dLdK2 = np.zeros((self.visible_size, self.visible_size))
            dLdL2 = np.zeros((self.visible_size, self.visible_size))


            # Likelihood accumulator
            log_ell = 0

            # State of b neurons at time [t-1]
            b_t1 = self.b_0

            # In t==1 we need the derivative of b wrt K and L at t-1,
            # and that'd require s_{t-2} which does not exist at that time step
            db_t1_dK = np.zeros((self.visible_size, self.visible_size, self.visible_size))
            db_t1_dL = np.zeros((self.visible_size, self.visible_size, self.visible_size))

            # We start in index 1 because we do not have s_{t-1} for t=0
            for t in range(1, T):

                # Compute the effective field of every neuron

                b = self.compute_b(s[t - 1], b_t1)

                h = self.compute_h(s[t - 1], b)
                tanh_h = np.tanh(h)
                sub_s_tanhh = s[t] - tanh_h

                # Compute the log Likelihood to check
                log_ell += np.dot(s[t], h) - np.sum(np.log(2 * np.cosh(h)))

                # Derivative of the Likelihood wrt H
                dLdH += sub_s_tanhh

                # Derivative of the Likelihood wrt J
                dLdJ += np.einsum('i,j->ij', sub_s_tanhh, s[t - 1])

                # if t == 1:
                #     # Compute the gradient of the Likelihood wrt b(0) at t==1
                #     dLdb_0 = np.dot(sub_s_tanhh, self.M)
                # if t == 2:
                #     # Compute the gradient of the Likelihood wrt b(0) at t==2
                #     b_t1_sq_rows = broadcast_rows((1 - b_t1 ** 2), self.visible_size)
                #     dLdb_0_2 = np.dot(sub_s_tanhh, np.einsum('ig,gz->iz', (self.M * b_t1_sq_rows), self.L))
                #
                #     dLdb_0 += dLdb_0_2

                # At t==1 b_t1_dK=0 and b_t1_dL=0
                # Derivative of b wrt K
                db_dK = np.einsum('ig,gnm->inm', self.L, db_t1_dK * (1 - np.tanh(b_t1) ** 2))
                # Derivative of b wrt L
                db_dL = np.einsum('ig,gnm->inm', self.L, db_t1_dL * (1 - np.tanh(b_t1) ** 2))
                for i in range(0, self.visible_size):
                    db_dK[i, i, :] += s[t - 1]
                    db_dL[i, i, :] += np.tanh(b_t1)

                # Compute the Jacobians
                # Derivative of the Likelihood wrt K
                dLdK += np.einsum('i,inm->nm', sub_s_tanhh,  db_dK)
                # Derivative of the Likelihood wrt L
                dLdL += np.einsum('i,inm->nm', sub_s_tanhh, db_dL)

                # Save the variables for the next step
                b_t1 = copy.deepcopy(b)
                db_t1_dK = copy.deepcopy(db_dK)
                db_t1_dL = copy.deepcopy(db_dL)

            # Normalize the gradients temporally and by the number of spins in the sum of the Likelihood
            dLdH /= self.visible_size * (T - 1)
            dLdJ /= self.visible_size * (T - 1)
            dLdK /= self.visible_size * (T - 1)
            dLdL /= self.visible_size * (T - 1)
            dLdb_0 /= self.visible_size * (T - 1)
            log_ell /= self.visible_size * (T - 1)

            error = self.gradient_descent(dLdH, dLdJ, dLdK, dLdL, dLdb_0, eta, mode=gradient_mode)

            if rep % plot_interval == 0:
                sim_s = self.simulate_hidden(T_sim, burn_in=100)
                sim_m, sim_C, sim_D = self.compute_moments(sim_s, T_sim)

                MSE_m = np.mean((m - sim_m) ** 2)
                MSE_C = np.mean((C - sim_C) ** 2)
                MSE_D = np.mean((D - sim_D) ** 2)

                # print('MSE m', MSE_m, 'C', MSE_C, 'D', MSE_D)
                # print()

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
        print('dLdb0', dLdb_0)

        return rep, ell_list, error_list, MSE_m_list, MSE_C_list, MSE_D_list, error_iter_list

    def simulate_full(self, T, burn_in=0):
        """
        Simulate the full Kinetic Ising model to produce data

        :param T: number of steps to simulate
        :param burn_in:
        :return:
        """
        full_s = []
        s = []
        for t in range(0, T + burn_in):
            self.ising.ParallelUpdate()
            full_s.append(self.ising.s)
            if t >= burn_in:
                s.append(self.ising.s[self.visible_idx])
        # print('Spins', s)
        return full_s, s

    # Update the state of the network using Little parallel update rule
    def hidden_parallel_update(self, sim_s_t1, b_t1):

        h = self.compute_h(sim_s_t1, b_t1)
        r = self.rng.random(self.visible_size)
        sim_s_t1 = -1 + 2 * (2 * self.Beta * h > - np.log(1 / r - 1)).astype(int)

        return sim_s_t1

    def simulate_hidden(self, T, burn_in=0):
        # Initialize all visible neurons to -1
        sim_s = np.ones(self.visible_size) * -1
        sim_s_list = []
        b_t1 = self.b_0

        for t in range(0, T + burn_in):

            b = self.compute_b(sim_s, b_t1)
            sim_s = self.hidden_parallel_update(sim_s, b_t1)
            if t >= burn_in:
                sim_s_list.append(sim_s)

            b_t1 = copy.deepcopy(b)

        return sim_s_list

    @staticmethod
    def compute_moments(s_list, T_ori):
        X = np.array(s_list).T
        m = X.mean(axis=1)
        C = X.dot(X.T) / T_ori
        C -= np.einsum('i,k->ik', m, m, optimize=True)
        d = 1
        D = X[:, 0:-d].dot(X[:, d:].T) / T_ori
        D -= np.einsum('i,k->ik', m, m, optimize=True)

        return m, C, D




class HiddenIsingg:  # Asymmetric Ising model simulation class with hidden activity

    def __init__(self, original_ising, visible_units_per):  # Create ising model

        self.ising = original_ising
        self.size = self.ising.size

        self.visible_size = int(self.size * visible_units_per)  # Network size
        self.hidden_size = self.size - self.visible_size

        # if b_size:
        #     self.b_size = b_size
        # else:
        #     self.b_size = self.hidden_size

        # self.H = np.zeros(self.visible_size)  # Fields
        self.J = np.zeros((self.visible_size, self.visible_size))  # Spin-to-Spin couplings
        self.K = np.zeros((self.visible_size, self.visible_size))  # Hidden-to-Neuron couplings
        self.L = np.zeros((self.visible_size, self.visible_size))  # Hidden-to-Hidden couplings

        # visible_units = np.ones(self.size)
        # self.hidden_idx = random.sample(range(0, self.ising.size), self.hidden_size)
        self.visible_idx = random.sample(range(0, self.ising.size), self.visible_size)
        # visible_units[hidden_idx] = 0

    def random_wiring(self):  # Set random values for J
        self.J = np.random.randn(self.visible_size, self.visible_size) / self.visible_size
        self.K = np.random.randn(self.visible_size, self.visible_size) / self.visible_size
        self.L = np.random.randn(self.visible_size, self.visible_size) / self.visible_size

    def sim_fit(self):

        # Number of time steps in the simulation
        T = 100
        # Simulate the full Kinetic Ising model to produce data
        full_s = []
        s = []
        for t in range(0, T):
            self.ising.ParallelUpdate()
            full_s.append(self.ising.s)
            s.append(self.ising.s[self.visible_idx])

        # Initialize variables for learning
        eta = 0.1
        rep = 0
        max_reps = 1000
        error_lim = 0.001
        error = np.inf
        # Learning loop
        while error > error_lim and rep < max_reps:
            # Initialize the gradients to 0
            LdJ = np.zeros((self.visible_size, self.visible_size))
            LdK = np.zeros((self.visible_size, self.visible_size))
            LdL = np.zeros((self.visible_size, self.visible_size))

            # State of b neurons at time [t-2]
            b = np.zeros(self.visible_size)
            b_t1 = np.zeros(self.visible_size)

            # We start in index 1 because we do not have s_{t-1} for t=0
            for t in range(1, T):

                if t != 1:
                    b = np.tanh(np.dot(self.K, s[t - 1]) + np.dot(self.L, b_t1))

                # Compute the derivative of the Likelihood wrt J
                tanh_h = np.tanh(b + np.dot(self.J, s[t - 1]))
                sub_s_h = s[t] - tanh_h

                LdJ += np.einsum('i,j->ij', sub_s_h, s[t - 1])

                # Compute the derivatives of b wrt L and K
                if t == 1:
                    # We need the derivative of b wrt K and L, and that'd require s_{t-2}
                    # Which does not exist at this time step
                    b_dK = np.zeros((self.visible_size, self.visible_size, self.visible_size))
                    b_dL = np.zeros((self.visible_size, self.visible_size, self.visible_size))
                else:
                    for i in range(0, self.visible_size):
                        # Derivative of b wrt K
                        for n in range(0, self.visible_size):
                            for m in range(0, self.visible_size):
                                # sub_b_1_sq = 1 - b ** 2
                                b_dK[i, n, m] = (np.dot(self.L[i, :], (b_t1_dK[:, n, m] * (1 - np.tanh(b) ** 2))))

                                if i == n:
                                    b_dK[i, n, m] += s[t - 2][m]

                        # Derivative of b wrt L
                        for n in range(0, self.visible_size):
                            for m in range(0, self.visible_size):
                                b_dL[i, n, m] = (np.dot(self.L[i, :], (b_t1_dL[:, n, m]) * (1 - np.tanh(b) ** 2)))

                                if i == n:
                                    b_dL[i, n, m] += b_t1[m]

                # Compute the Jacobians
                # Derivative of Likelihood wrt K
                for n in range(0, self.visible_size):
                    for m in range(0, self.visible_size):
                        LdK[n, m] += np.dot(sub_s_h, b_dK[:, n, m])

                # Derivative of Likelihood wrt L
                for n in range(0, self.visible_size):
                    for m in range(0, self.visible_size):
                        LdL[n, m] += np.dot(sub_s_h, b_dL[:, n, m])

                b_t1 = copy.deepcopy(b)
                b_t1_dK = copy.deepcopy(b_dK)
                b_t1_dL = copy.deepcopy(b_dL)

            # Normalize the gradients temporally and by the number of spins
            LdJ /= (self.visible_size * T)
            LdK /= (self.visible_size * T)
            LdL /= (self.visible_size * T)

            self.J = self.J + eta * LdJ
            self.K = self.K + eta * LdK
            self.L = self.L + eta * LdL

            error = max(np.max(np.abs(LdJ)), np.max(np.abs(LdK)), np.max(np.abs(LdL)))

            print(rep)
            print(error)

            rep = rep + 1


if __name__ == "__main__":
    kinetic_ising = ising(netsize=10)
    hidden_ising = HiddenIsing(kinetic_ising, visible_units_per=0.6, b_size=2)
    hidden_ising.random_wiring()

    hidden_ising.sim_fit()
