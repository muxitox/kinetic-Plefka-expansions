#!/usr/bin/env python3
"""
GPLv3 2020 Miguel Aguilera

This code allows to run simulations of the kinetic Ising model,
with asymmetric weights and parallel updates.
"""

import numpy as np

class ising:              # Asymmetric Ising model simulation class

    def __init__(self, netsize, rng=None):  # Create ising model

        self.size = netsize                        # Network size
        self.H = np.zeros(netsize)                # Fields
        self.J = np.zeros((netsize, netsize))  # Couplings
        self.Beta = 1                            # Inverse temperature

        if rng:
            self.rng = rng
        else:
            self.rng = np.random.default_rng()
        self.randomize_state()                    # Set random state

    def randomize_state(self):        # Randomize network state
        self.s = self.rng.integers(0, 2, self.size) * 2 - 1

    def random_fields(self):        # Set random values for H
        self.H = self.rng.normal(loc=0.0, scale=1, size=self.size) * 2 - 1

    def random_wiring(self):  # Set random values for J
        self.J = self.rng.normal(loc=0.0, scale=1, size=(self.size, self.size)) / self.size
        # self.J = self.rng.random((self.size, self.size)) / np.sqrt(self.size)
        # self.J = 1/self.size + self.rng.random((self.size, self.size)) / np.sqrt(self.size)
        # self.J = np.ones((self.size, self.size)) * 1/self.size


    # Update the state of the network using Little parallel update rule
    def ParallelUpdate(self):
        self.h = self.H + np.dot(self.J, self.s)
        r = self.rng.random(self.size)
        self.s = -1 + 2 * (2 * self.Beta * self.h > -
                           np.log(1 / r - 1)).astype(int)
