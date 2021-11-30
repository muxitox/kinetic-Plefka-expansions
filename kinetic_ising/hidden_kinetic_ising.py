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

        R = 100
        T = 100

        derJ = 0
        derK = 0
        derL = 0

        for t in range(0,T):
            for r in range(0,R):
                self.ising.ParallelUpdate()
                s = self.ising.s

                if r != 0:

                    self.b = np.tanh(np.dot(self.K, s_p) + np.dot(self.L*b_p))
                    tanh_h = np.tanh(self.b + np.dot(self.J, s_p))
                    # Compute quantities for the gradients
                    sub_s_h = s - tanh_h

                    derJ += np.dot(sub_s_h, s_p)


                s_p = copy.deepcopy(s)
                b_p = copy.deepcopy(self.b)

if __name__=="__main__":

    kinetic_ising = ising(netsize=10)
    hidden_ising = HiddenIsing(kinetic_ising, visible_units_per=0.7, b_size=None)

