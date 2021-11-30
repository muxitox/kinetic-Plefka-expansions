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

        self.visible_size = self.size * visible_units_per  # Network size
        self.hidden_size = self.size - self.visible_size
        self.H = np.zeros(self.visible_size)  # Fields
        self.J = np.zeros((self.visible_size, self.visible_size))  # Spin-to-Spin couplings
        self.K = np.zeros((self.visible_size, self.visible_size))  # Hidden-to-Neuron couplings
        self.L = np.zeros((self.visible_size, self.visible_size))  # Hidden-to-Hidden couplings

        if b_size:
            self.b_size = b_size
        else:
            self.b_size = self.hidden_size

        self.b = np.zeros(self.b_size)  # Fields

        visible_units = np.ones(self.size)
        hidden_idx = random.sample(range(0, self.ising.size), self.hidden_size)
        visible_units[hidden_idx] = 0

    def random_fields(self):  # Set random values for H
        self.H = np.random.rand(self.size) * 2 - 1

    def random_wiring(self):  # Set random values for J
        self.J = np.random.randn(self.size, self.size) / self.size
        self.K = np.random.randn(self.size, self.size) / self.size
        self.L = np.random.randn(self.size, self.size) / self.size


    def sim_fit(self):

        T = 100
        # Produce the data:
        s = []
        for t in range(0,T):
            self.ising.ParallelUpdate()
            s.append(self.ising.s)

        LderJ_accum = 0
        LderK_accum = 0
        LderL_accum = 0

        rep = 0
        max_reps = 200
        error_lim = 0.0001
        error = np.inf
        while error > error_lim:
            for t in range(0,T):
                if t != 0:

                    # Compute quantities for the gradients
                    self.b = np.tanh(np.dot(self.K, s[t-1]) + np.dot(self.L*b_p))
                    tanh_h = np.tanh(self.b + np.dot(self.J, s[t-1]))
                    sub_s_h = s[t] - tanh_h

                    LderJ_accum += np.dot(sub_s_h, s[t-1])

                    if r == 1:

                    else:

                b_p = copy.deepcopy(self.b)


            rep = rep + 1
            if rep > max_reps:
                break


if __name__=="__main__":

    kinetic_ising = ising(netsize=10)
    hidden_ising = HiddenIsing(kinetic_ising, visible_units_per=0.7, b_size=None)

