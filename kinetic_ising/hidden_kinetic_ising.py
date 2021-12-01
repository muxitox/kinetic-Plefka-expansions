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
            LdJ = 0
            LdK = 0
            LdL = 0

            b_p = np.zeros(self.b_size)  # Variable fields Fields

            # We start in index 1 because we dont have s_{t-1} for t=0
            for t in range(1, T):

                # Compute the derivative of the Likelihood wrt J
                self.b = np.tanh(np.dot(self.K, s[t-1]) + np.dot(self.L, b_p))
                tanh_h = np.tanh(self.b + np.dot(self.J, s[t-1]))
                sub_s_h = s[t] - tanh_h
                LdJ += np.einsum('i,j->ij', sub_s_h, s[t-1])

                # Compute the derivatives of the Likelihood wrt L and K
                if t == 1:
                    # We need the derivative of b wrt K and L, and that'd require s_{t-2}
                    # Which does not exist at this time step
                    bdK_p = 0
                    bdL_p = 0
                else:
                    sub_b_1_sq = 1 - b_p ** 2
                    bdK_p = np.dot(s[t-2] + np.dot(self.L, bdK_p2), sub_b_1_sq)
                    bdL_p = np.dot(s[t-2] + np.dot(self.L, bdL_p2), sub_b_1_sq)

                sub_b_sq = 1 - self.b**2
                LdK += np.dot(np.dot(sub_s_h, (s[t-1] + np.dot(self.L, bdK_p))), sub_b_sq)
                LdL += np.dot(np.dot(sub_s_h, (b_p + np.dot(self.L, bdL_p))), sub_b_sq)

            b_p = copy.deepcopy(self.b)
            bdK_p2 = copy.deepcopy(bdK_p)
            bdL_p2 = copy.deepcopy(bdL_p)

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

