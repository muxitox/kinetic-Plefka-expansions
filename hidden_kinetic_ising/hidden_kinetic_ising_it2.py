#!/usr/bin/env python3
"""

This code allows to run simulations of the kinetic Ising model,
with asymmetric weights and parallel updates.
"""
import time

import numpy as np
from kinetic_ising import ising
import copy
from utils import *
import matplotlib.pyplot as plt


class HiddenIsing:  # Asymmetric Ising model with hidden activity simulation class

    def __init__(self, original_ising, visible_units_per, b_size=0, rng=None):  # Create ising model
        """
        Initializes the class for simulation

        :param original_ising: ising model you want to learn from
        :param visible_units_per: percentage of units that are visible
        :param b_size: number of b type "hidden" neurons
        :param rng: random number generator. If not set, one is created.
        """

        self.ising = original_ising
        self.size = self.ising.size

        self.visible_size = int(self.size * visible_units_per)      # Network size
        self.hidden_size = self.size - self.visible_size
        self.b_size = b_size

        self.J = np.zeros((self.visible_size, self.visible_size))   # Spin-to-Spin couplings
        self.M = np.zeros((self.visible_size, self.b_size))         # Hidden-to-Hidden couplings
        self.K = np.zeros((self.b_size, self.visible_size))         # Hidden-to-Neuron couplings
        self.L = np.zeros((self.b_size, self.b_size))               # Hidden-to-Hidden couplings
        self.b_0 = np.zeros(self.b_size)

        self.Beta = 1                                               # Inverse temperature


        if rng:
            self.rng = rng
        else:
            self.rng = np.random.default_rng()

        # visible_units = np.ones(self.size)
        # self.hidden_idx = random.sample(range(0, self.ising.size), self.hidden_size)
        self.visible_idx = self.rng.choice(range(0, self.ising.size), self.visible_size)
        # visible_units[hidden_idx] = 0

    def random_wiring(self):  # Set random values for J
        self.J = self.rng.random((self.visible_size, self.visible_size)) / self.visible_size
        self.M = self.rng.random((self.visible_size, self.b_size)) / ((self.visible_size + self.b_size) / 2)
        self.K = self.rng.random((self.b_size, self.visible_size)) / ((self.visible_size + self.b_size) / 2)
        self.L = self.rng.random((self.b_size, self.b_size)) / self.b_size
        self.b_0 = self.rng.random(self.b_size) * 2 - 1

        print('J', self.J)
        print('M', self.M)
        print('K', self.K)
        print('L', self.L)
        print()

    def gradient_descent(self, dLdJ, dLdK, dLdL, dLdM, dLdb_0, eta, mode='regular'):

        if mode == 'regular':
            error = self.regular_gradient_descent(dLdJ, dLdK, dLdL, dLdM, dLdb_0, eta)
        elif mode == 'coordinated':
            error = self.coordinated_gradient_descent(dLdJ, dLdK, dLdL, dLdM, dLdb_0, eta)

        return error

    def regular_gradient_descent(self, dLdJ, dLdK, dLdL, dLdM, dLdb_0, eta):

        if b_size > 0:

            error = max(np.max(np.abs(dLdJ)), np.max(np.abs(dLdM)), np.max(np.abs(dLdK)), np.max(np.abs(dLdL)),
                        np.max(dLdb_0))

            self.J = self.J + eta * dLdJ
            self.M = self.M + eta * dLdM
            self.K = self.K + eta * dLdK
            self.L = self.L + eta * dLdL
            self.b_0 = self.b_0 + eta * dLdb_0
        else:
            error = np.max(np.abs(dLdJ))

            self.J = self.J + eta * dLdJ

        return error

    def coordinated_gradient_descent(self, dLdJ, dLdK, dLdL, dLdM, dLdb_0, eta):

        if self.b_size > 0:
            max_J_idx = np.argmax(np.abs(dLdJ))
            max_J_idx = np.unravel_index(max_J_idx, dLdJ.shape)
            max_M_idx = np.argmax(np.abs(dLdM))
            max_M_idx = np.unravel_index(max_M_idx, dLdM.shape)
            max_K_idx = np.argmax(np.abs(dLdK))
            max_K_idx = np.unravel_index(max_K_idx, dLdK.shape)
            max_L_idx = np.argmax(np.abs(dLdL))
            max_L_idx = np.unravel_index(max_L_idx, dLdL.shape)
            max_b0_idx = np.argmax(np.abs(dLdb_0))
            max_b0_idx = np.unravel_index(max_b0_idx, dLdb_0.shape)

            max_J = dLdJ[max_J_idx]
            max_M = dLdM[max_M_idx]
            max_K = dLdK[max_K_idx]
            max_L = dLdL[max_L_idx]
            max_b0 = dLdb_0[max_b0_idx]

            max_max = np.argmax(np.abs([max_J, max_M, max_K, max_L, max_b0]))
            if max_max == 0:
                self.J[max_J_idx] = self.J[max_J_idx] + eta * max_J
                error = np.abs(max_J)

            elif max_max == 1:
                self.M[max_M_idx] = self.M[max_M_idx] + eta * max_M
                error = np.abs(max_M)

            elif max_max == 2:
                self.K[max_K_idx] = self.K[max_K_idx] + eta * max_K
                error = np.abs(max_K)

            elif max_max == 3:
                self.L[max_L_idx] = self.L[max_L_idx] + eta * max_L
                error = np.abs(max_L)


            elif max_max == 4:
                self.b_0[max_b0_idx] = self.b_0[max_b0_idx] + eta * max_b0
                error = np.abs(max_b0)

        else:
            max_J_idx = np.argmax(np.abs(dLdJ))
            max_J_idx = np.unravel_index(max_J_idx, dLdJ.shape)
            self.J[max_J_idx] = self.J[max_J_idx] + eta * dLdJ[max_J_idx]
            error = np.abs(dLdJ[max_J_idx])

        return error

    def compute_b(self, s_t1, b_t1):
        """

        :param s_t1: State of the spins at time t-1
        :param b_t1: State of the hidden neurons at time t-1
        :return: state of the hidden neurons at time t
        """
        return np.tanh(np.dot(self.K, s_t1) + np.dot(self.L, b_t1))

    def compute_h(self, s_t1, b_t1):
        """

        :param s_t1: State of the spins at time t-1
        :param b_t1: State of the hidden neurons at time t-1
        :return: effective field at time t
        """

        return np.dot(self.M, b_t1) + np.dot(self.J, s_t1)



    def fit(self, s, eta, max_reps):

        """

        :param s: evolution of the system trough T steps
        :return:
        """


        # Initialize variables for learning
        rep = 0
        error_lim = 0.0005
        error = np.inf
        # Learning loop
        old_error_L = np.inf
        old_error_b0 = np.inf
        old_error_K = np.inf
        old_error_M = np.inf
        old_error_J = np.inf
        T = len(s)

        ell_list = np.zeros(max_reps)
        error_list = np.zeros(max_reps)

        old_error = np.inf

        log_ell_t1 = -np.inf

        while error > error_lim and rep < max_reps:
            if rep % 100==0:
                print('Iter', rep)

            # Initialize the gradients to 0
            dLdJ = np.zeros((self.visible_size, self.visible_size))
            dLdM = np.zeros((self.visible_size, self.b_size))
            dLdK = np.zeros((self.b_size, self.visible_size))
            dLdL = np.zeros((self.b_size, self.b_size))
            dLdb_0 = np.zeros(self.b_size)

            # Likelihood accumulator
            log_ell = 0

            # State of b neurons at time [t-1]
            b_t1 = self.b_0

            # In t==1 we need the derivative of b wrt K and L at t-1,
            # and that'd require s_{t-2} which does not exist at that time step
            b_t1_dK = np.zeros((self.b_size, self.b_size, self.visible_size))
            b_t1_dL = np.zeros((self.b_size, self.b_size, self.b_size))

            # db_dK = np.zeros((self.b_size, self.b_size, self.visible_size))
            # db_dL = np.zeros((self.b_size, self.b_size, self.b_size))

            # We start in index 1 because we do not have s_{t-1} for t=0
            for t in range(1, T):

                # Compute the derivative of the Likelihood wrt J
                h = self.compute_h(s[t - 1], b_t1)
                tanh_h = np.tanh(h)
                # print('h', np.dot(self.M, b_t1) + np.dot(self.J, s[t - 1]))
                # print('tanh_h', tanh_h)
                sub_s_tanhh = s[t] - tanh_h
                # print('sub_s_tanhh', sub_s_tanhh)

                # Compute the log Likelihood to check
                log_ell += np.dot(s[t], h) - np.sum(np.log(2 * np.cosh(h)))

                # Derivative of the Likelihood wrt J
                dLdJ += np.einsum('i,j->ij', sub_s_tanhh, s[t - 1])

                # Save computational load if the number of b neurons < 1
                if self.b_size > 0:
                    dLdM += np.einsum('i,j->ij', sub_s_tanhh, b_t1)

                    if t == 1:
                        # Compute the gradient of the Likelihood wrt b(0) at t==1
                        dLdb_0 = np.dot(sub_s_tanhh, self.M)
                    if t == 2:
                        # Compute the gradient of the Likelihood wrt b(0) at t==2
                        b_t1_sq_rows = broadcast_rows((1 - b_t1**2), self.visible_size)
                        dLdb_0_2 = np.dot(sub_s_tanhh, np.einsum('ig,gz->iz', (self.M*b_t1_sq_rows), self.L))

                        dLdb_0 += dLdb_0_2

                    # Derivative of the Likelihood wrt K
                    dLdK += np.einsum('i,inm->nm', sub_s_tanhh, np.einsum('ig,gnm->inm', self.M, b_t1_dK))
                    # Derivative of the Likelihood wrt L
                    dLdL += np.einsum('i,inm->nm', sub_s_tanhh, np.einsum('ig,gnm->inm', self.M, b_t1_dL))

                    # Compute the necessary information for the next step
                    # At t==1, b(t-1)=0
                    b = self.compute_b(s[t - 1], b_t1)
                    # print('b', b)

                    # At t==1 b_t1_dK=0 and b_t1_dL=0
                    # Derivative of b wrt K
                    db_dK = np.einsum('ig,gnm->inm', self.L, b_t1_dK)
                    # Derivative of b wrt L
                    db_dL = np.einsum('ig,gnm->inm', self.L, b_t1_dL)
                    for i in range(0, self.b_size):
                        db_dK[i, i, :] += s[t - 1]
                        db_dK[i] *= (1 - b[i] ** 2)

                        db_dL[i, i, :] += b_t1
                        db_dL[i] *= (1 - b[i] ** 2)

                    # Save the variables for the next step
                    b_t1 = copy.deepcopy(b)
                    b_t1_dK = copy.deepcopy(db_dK)
                    b_t1_dL = copy.deepcopy(db_dL)

            # Normalize the gradients temporally and by the number of spins in the sum of the Likelihood
            dLdJ /= self.visible_size * (T - 1)
            dLdM /= self.visible_size * (T - 1)
            dLdK /= self.visible_size * (T - 1)
            dLdL /= self.visible_size * (T - 1)
            dLdb_0 /= self.visible_size * (T - 1)
            log_ell /= self.visible_size * (T - 1)

            error = self.gradient_descent(dLdJ, dLdK, dLdL, dLdM, dLdb_0, eta, mode='regular')


            # print('b0', self.b_0)
            # print('L', self.L)

            # Prints for debugging
            # if self.b_size > 1:
            #     self.L[1][1] = self.L[1][1] + eta * LdL[1][1]
            # else:
            #     self.L[0] = self.L[0] + eta * LdL[0]

            # print('J', self.J)
            # print('M', self.M)
            # print('b0', self.b_0)
            # print('K', self.K)
            # print('L', self.L)
            # print()

            error_list[rep] = error
            # print('max_error', error)

            # print('log Likelihood', log_ell)

            if log_ell_t1 > log_ell:
                print('#################################### WRONG | LIKELIHOOD DECREASING')

            # print('Comparison', '(log_ell-log_ell_t1)/eta', (log_ell-log_ell_t1)/eta, 'dLdb_0²', old_error_b0**2)

            ell_list[rep] = log_ell
            log_ell_t1 = log_ell

            # if self.b_size > 1:
            #     print('dLdL', dLdL)
            #     print('dLdM', dLdM)
            #     print('dLdJ', dLdJ)
            #     print('dLdK', dLdK)
            #     print('dLdb0', dLdb_0)

                # if np.abs(dLdL[1][1]) - np.abs(old_error_L) > 0:
                #     print('#################################### WRONG | L GRADIENT MAGNITUDE INCREASING')
                # old_error_L = dLdL[1][1]

            # else:
            #     print('dLdL', dLdL)
            #     print('dLdM', dLdM)
            #
            #     print('dLdJ', dLdJ)
            #     print('dLdK', dLdK)
            #     print('dLdb0', dLdb_0)

                # if np.abs(LdL[0]) > np.abs(old_error_L) > 0:
                #     print('#################################### WRONG | L GRADIENT MAGNITUDE INCREASING')

                # if np.abs(dLdM[0]) > np.abs(old_error_M):
                #     print('#################################### WRONG | M GRADIENT MAGNITUDE INCREASING')
                #
                # if np.abs(dLdJ[0]) > np.abs(old_error_J):
                #     print('#################################### WRONG | J GRADIENT MAGNITUDE INCREASING')
                #
                # if np.abs(LdK[0]) > np.abs(old_error_K):
                #     print('#################################### WRONG | K GRADIENT MAGNITUDE INCREASING')

                # if np.abs(dLdb_0) > np.abs(old_error_b0):
                #     print('#################################### WRONG | b GRADIENT MAGNITUDE INCREASING')
                # old_error_L = dLdL[0]
                # old_error_b0 = dLdb_0
                # old_error_K = dLdK[0]
                # old_error_M = dLdM[0]
                # old_error_J = dLdJ[0]

            rep = rep + 1

            # print()

        print('dLdL', dLdL)
        print('dLdM', dLdM)

        print('dLdJ', dLdJ)
        print('dLdK', dLdK)
        print('dLdb0', dLdb_0)

        return rep, ell_list, error_list


    def simulate_full(self, T):
        """
        Simulate the full Kinetic Ising model to produce data

        :param T: number of steps to simulate
        :return:
        """
        full_s = []
        s = []
        for t in range(0, T):
            self.ising.ParallelUpdate()
            full_s.append(self.ising.s)
            s.append(self.ising.s[self.visible_idx])
        # print('Spins', s)
        return full_s, s

    # Update the state of the network using Little parallel update rule
    def parallel_update(self, sim_s, b_t1):

        h = self.compute_h(sim_s, b_t1)
        r = self.rng.random(self.visible_size)
        sim_s = -1 + 2 * (2 * self.Beta * h > -
        np.log(1 / r - 1)).astype(int)

        return sim_s

    def simulate_hidden(self, T):
        # Initialize all visible neurons to -1
        sim_s = np.ones(self.visible_size) * -1
        sim_s_list = []
        b_t1 = self.b_0

        for t in range(0, T):

            b = self.compute_b(sim_s, b_t1)

            sim_s = self.parallel_update(sim_s, b_t1)
            sim_s_list.append(sim_s)

            b_t1 = copy.deepcopy(b)

        return sim_s_list

    @staticmethod
    def compute_moments(s_list):
        X = np.array(s_list).T
        m = X.mean(axis=1)
        C = X.dot(X.T) / T
        C -= np.einsum('i,k->ik', m, m, optimize=True)
        d = 1
        D = X[:, 0:-d].dot(X[:, d:].T) / T
        D -= np.einsum('i,k->ik', m, m, optimize=True)

        return m, C, D


if __name__ == "__main__":
    # You can set up a seed here for reproducibility
    # Seed to check wrong behavior: 6, 2425, 615
    # 3656, 0.6 1, 2, 3

    reproducible = True
    if reproducible:
        seed = [3813]
    else:
        seed = np.random.randint(5000, size=1)

    rng = np.random.default_rng(seed)

    print('Seed', seed)

    kinetic_ising = ising(netsize=10, rng=rng)
    kinetic_ising.random_fields()
    kinetic_ising.random_wiring()
    vis_per = 0.6
    b_size = 3
    hidden_ising = HiddenIsing(kinetic_ising, visible_units_per=vis_per, b_size=b_size, rng=rng)
    hidden_ising.random_wiring()

    T = 100
    eta = 0.01
    max_reps = 60000
    full_s, visible_s = hidden_ising.simulate_full(T)
    m, C, D = hidden_ising.compute_moments(visible_s)
    num_reps, ell_list, error_list = hidden_ising.fit(visible_s, eta, max_reps)

    plt.plot(ell_list[0:num_reps], label='log(ell)')
    plt.plot(error_list[0:num_reps], label='max_grad')
    plt.plot(np.square(error_list[0:num_reps]), label='sq_max_grad')
    plt.plot(np.diff(ell_list[0:num_reps]/eta), '--', label='np.diff(log_ell)/eta')

    plt.xlabel('iters')
    title_str = f'Seed: {seed}. Vis_units: {vis_per}. b_size: {b_size} Simulation steps: {T}. eta: {eta}.'
    plt.title(title_str)
    plt.legend()
    plt.show()

    sim_s = hidden_ising.simulate_hidden(T)
    sim_m, sim_C, sim_D = hidden_ising.compute_moments(sim_s)

    print(title_str)
    print('Error m', m - sim_m)
    print()
    print('Error C', C - sim_C)
    print()
    print('Error D',  D - sim_D)

    print('Seed', seed)
