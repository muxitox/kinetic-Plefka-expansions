#!/usr/bin/env python3
"""

This code allows to run simulations of the kinetic Ising model,
with asymmetric weights and parallel updates.
"""

import numpy as np
from kinetic_ising import ising
import random
import copy


class HiddenIsing:  # Asymmetric Ising model simulation class with hidden activity

    def __init__(self, original_ising, visible_units_per, b_size=None):  # Create ising model

        self.ising = original_ising
        self.size = self.ising.size

        self.visible_size = int(self.size * visible_units_per)  # Network size
        self.hidden_size = self.size - self.visible_size

        if b_size:
            self.b_size = b_size
        else:
            self.b_size = self.hidden_size

        # self.H = np.zeros(self.visible_size)  # Fields
        self.J = np.zeros((self.visible_size, self.visible_size))  # Spin-to-Spin couplings
        self.K = np.zeros((self.visible_size, self.visible_size))  # Hidden-to-Neuron couplings
        self.L = np.zeros((self.visible_size, self.b_size))  # Hidden-to-Hidden couplings
        self.b = np.zeros(self.b_size)




        # visible_units = np.ones(self.size)
        # self.hidden_idx = random.sample(range(0, self.ising.size), self.hidden_size)
        self.visible_idx = random.sample(range(0, self.ising.size), self.visible_size)
        # visible_units[hidden_idx] = 0

    def random_wiring(self):  # Set random values for J
        self.J = np.random.randn(self.visible_size, self.visible_size) / self.visible_size
        self.K = np.random.randn(self.visible_size, self.visible_size) / self.visible_size
        self.L = np.random.randn(self.visible_size, self.b_size) / self.visible_size


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
        max_reps = 200
        error_lim = 0.0001
        error = np.inf
        # Learning loop
        while error > error_lim and rep < max_reps:
            # Initialize the gradients to 0
            LdJ = np.zeros((self.visible_size, self.visible_size))
            LdK = np.zeros((self.visible_size, self.visible_size))
            LdL = np.zeros((self.visible_size, self.b_size))

            b_t1 = np.zeros(self.b_size)  # Variable fields Fields

            # We start in index 1 because we dont have s_{t-1} for t=0
            for t in range(1, T):

                # Compute the derivative of the Likelihood wrt J
                self.b = np.tanh(np.dot(self.K, s[t-1]) + np.dot(self.L, b_t1))
                tanh_h = np.tanh(self.b + np.dot(self.J, s[t-1]))
                sub_s_h = s[t] - tanh_h
                LdJ += np.einsum('i,j->ij', sub_s_h, s[t-1])

                # Compute the derivatives of the Likelihood wrt L and K
                if t == 1:
                    # We need the derivative of b wrt K and L, and that'd require s_{t-2}
                    # Which does not exist at this time step
                    b_t1_dK = np.zeros((self.b_size, self.visible_size, self.visible_size))
                    b_t1_dL = np.zeros((self.b_size, self.visible_size, self.b_size))
                else:
                    for i in range(0, self.b_size):
                        for n in range(0, self.visible_size):
                            for m in range(0, self.visible_size):
                                # sub_b_1_sq = 1 - b_t1 ** 2
                                b_t1_dK[i, n, m] = (s[t-2][i] + np.dot(self.L[i, :], b_t2_dK[:, n, m])) * \
                                                   (1 - b_t1[i]**2)

                        for n in range(0, self.visible_size):
                            for m in range(0, self.b_size):
                                b_t1_dL[i, n, m] = (b_t2[i] + np.dot(self.L[i, :], b_t2_dL[:, n, m])) * \
                                                   (1 - b_t1[i] ** 2)

                    # b_t1_dK = np.einsum('i,k->ik', sub_b_1_sq, s[t-2] + np.dot(self.L, b_t2_dK))
                    # b_t1_dL = np.einsum('i,k->ik', sub_b_1_sq, s[t-2] + np.dot(self.L, b_t2_dL))

                # sub_b_sq = 1 - self.b**2
                # LdK += np.dot(np.dot(sub_s_h, (s[t-1] + np.dot(self.L, b_t1_dK))), sub_b_sq)
                # LdK += np.dot(np.dot(sub_s_h, (s[t-1] + np.dot(self.L, b_t1_dK))), sub_b_sq)

                # LdL += np.dot(np.dot(sub_s_h, (b_t1 + np.dot(self.L, b_t1_dL))), sub_b_sq)

                for i in range(0, self.visible_size):

                    left_product = (s[t][i] - tanh_h[i]) * (1 - self.b[i]**2)

                    for n in range(0, self.visible_size):
                        for m in range(0, self.visible_size):

                            LdK[n, m] = left_product * \
                                (s[t-1][m] + np.dot(self.L[i, :], b_t1_dK[:, n, m]))

                    for n in range(0, self.visible_size):
                        for m in range(0, self.b_size):
                            LdL[n, m] = left_product * \
                                (b_t1[m] + np.dot(self.L[i, :], b_t1_dL[:, n, m]))

                b_t2 = copy.deepcopy(b_t1)
                b_t1 = copy.deepcopy(self.b)
                b_t2_dK = copy.deepcopy(b_t1_dK)
                b_t2_dL = copy.deepcopy(b_t1_dL)

            self.J = self.J + eta * LdJ
            self.K = self.K + eta * LdK
            self.L = self.L + eta * LdL

            error = max(np.max(np.abs(LdJ)), np.max(np.abs(LdK)), np.max(np.abs(LdL)))

            rep = rep + 1


if __name__ == "__main__":

    kinetic_ising = ising(netsize=10)
    hidden_ising = HiddenIsing(kinetic_ising, visible_units_per=0.7, b_size=1)
    hidden_ising.random_wiring()

    hidden_ising.sim_fit()

