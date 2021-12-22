#!/usr/bin/env python3
"""

This code allows to run simulations of the kinetic Ising model,
with asymmetric weights and parallel updates.
"""

import numpy as np
from kinetic_ising import ising
import copy


class HiddenIsing:  # Asymmetric Ising model simulation class with hidden activity

    def __init__(self, original_ising, visible_units_per, b_size=None, rng=None):  # Create ising model

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
        self.M = np.zeros((self.visible_size, self.b_size))  # Hidden-to-Hidden couplings
        self.K = np.zeros((self.b_size, self.visible_size))  # Hidden-to-Neuron couplings
        self.L = np.zeros((self.b_size, self.b_size))  # Hidden-to-Hidden couplings

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

        print('J', self.J)
        print('M', self.M)
        print('K', self.K)
        print('L', self.L)
        print()

    def sim_fit(self):

        # Number of time steps in the simulation
        T = 200
        # Simulate the full Kinetic Ising model to produce data
        full_s = []
        s = []
        for t in range(0, T):
            self.ising.ParallelUpdate()
            full_s.append(self.ising.s)
            s.append(self.ising.s[self.visible_idx])

        # print('Spins', s)

        # Initialize variables for learning
        eta = 0.1
        rep = 0
        max_reps = 2000
        error_lim = 0.001
        error = np.inf
        # Learning loop
        old_error = np.inf
        while error > error_lim and rep < max_reps:
            # Initialize the gradients to 0
            LdJ = np.zeros((self.visible_size, self.visible_size))
            LdM = np.zeros((self.visible_size, self.b_size))
            LdK = np.zeros((self.b_size, self.visible_size))
            LdL = np.zeros((self.b_size, self.b_size))

            # State of b neurons at time [t-1]
            b_t1 = np.zeros(self.b_size)

            # In t==1 we need the derivative of b wrt K and L at t-1,
            # and that'd require s_{t-2} which does not exist at that time step
            b_t1_dK = np.zeros((self.b_size, self.b_size, self.visible_size))
            b_t1_dL = np.zeros((self.b_size, self.b_size, self.b_size))

            b_dK = np.zeros((self.b_size, self.b_size, self.visible_size))
            b_dL = np.zeros((self.b_size, self.b_size, self.b_size))


            # We start in index 1 because we do not have s_{t-1} for t=0
            for t in range(1, T):

                # Compute the derivative of the Likelihood wrt J
                tanh_h = np.tanh(np.dot(self.M, b_t1) + np.dot(self.J, s[t - 1]))
                # print('h', np.dot(self.M, b_t1) + np.dot(self.J, s[t - 1]))
                # print('tanh_h', tanh_h)
                sub_s_tanhh = s[t] - tanh_h
                # print('sub_s_tanhh', sub_s_tanhh)

                LdJ += np.einsum('i,j->ij', sub_s_tanhh, s[t - 1])

                # Save computational load if the number of b neurons < 1
                if self.b_size > 0:
                    LdM += np.einsum('i,j->ij', sub_s_tanhh, b_t1)

                    # Compute the Jacobians wrt K and L
                    for i in range(0, self.visible_size):
                        # Derivative of Likelihood wrt K
                        for n in range(0, self.b_size):
                            for m in range(0, self.visible_size):
                                LdK[n, m] += sub_s_tanhh[i] * \
                                            np.dot(self.M[i, :], b_t1_dK[:, n, m])

                        # Derivative of Likelihood wrt L
                        for n in range(0, self.b_size):
                            for m in range(0, self.b_size):
                                LdL[n, m] += sub_s_tanhh[i] * \
                                            np.dot(self.M[i, :], b_t1_dL[:, n, m])
                                # print('LdL[i,n,m,t]', i, n, m, t, sub_s_tanhh[i] * \
                                #             np.dot(self.M[i, :], b_t1_dL[:, n, m]))
                                # print('LdL[n,m] accum',  n, m, t, LdL[n, m])
                                # print('sub_s_tanhh[i]', sub_s_tanhh[i])

                    # Compute the necessary information for the next step
                    # At t==1, b(t-1)=0
                    b = np.tanh(np.dot(self.K, s[t - 1]) + np.dot(self.L, b_t1))
                    # print('b', b)

                    # Compute the derivatives of b wrt L and K
                    # At t==1 b_t1_dK=0 and b_t1_dL=0
                    for i in range(0, self.b_size):
                        # Derivative of b wrt K
                        for n in range(0, self.b_size):
                            for m in range(0, self.visible_size):
                                b_dK[i, n, m] = np.dot(self.L[i, :], b_t1_dK[:, n, m])
                                if i == n:
                                    b_dK[i, n, m] += s[t - 1][m]

                                b_dK[i, n, m] *= (1 - b[i] ** 2)

                        # Derivative of b wrt L
                        for n in range(0, self.b_size):
                            for m in range(0, self.b_size):

                                b_dL[i, n, m] = np.dot(self.L[i, :], b_t1_dL[:, n, m])
                                if i == n:
                                    b_dL[i, n, m] += b_t1[m]

                                b_dL[i, n, m] *= (1 - b[i] ** 2)
                                # print('b_dL[i,n,m](t)', i, n, m, t, b_dL[i, n, m])

                    b_t1 = copy.deepcopy(b)
                    b_t1_dK = copy.deepcopy(b_dK)
                    b_t1_dL = copy.deepcopy(b_dL)

            # Normalize the gradients temporally and by the number of spins in the sum of the Likelihood
            LdJ /= (T-1)
            LdM /= (T-1)
            LdK /= (self.visible_size * (T-1))
            LdL /= (self.visible_size * (T-1))

            # self.J = self.J + eta * LdJ
            # self.K = self.K + eta * LdK
            # self.L = self.L + eta * LdL
            # self.M = self.M + eta * LdM



            # Prints for debugging
            if self.b_size > 1:
                self.L[1][1] = self.L[1][1] + eta * LdL[1][1]
            else:
                self.L[0] = self.L[0] + eta * LdL[0]


            # print('J', self.J)
            # print('M', self.M)
            # print('K', self.K)
            # print('L', self.L)
            # print()

            # Compute the error as the max component of the gradients
            # if self.b_size > 1:
            #     error = max(np.max(np.abs(LdJ)), np.max(np.abs(LdM)), np.max(np.abs(LdK)), np.max(np.abs(LdL)))
            # else:
            #     error = np.max(np.abs(LdJ))


            print(rep)
            # print(error)
            if self.b_size>1:
                print('LdL[1][1]', LdL[1][1])

                if np.abs(LdL[1][1]) - np.abs(old_error) > 0:
                    print('#################################### WRONG | GRADIENT INCREASING')
                old_error = LdL[1][1]

            else:
                print('LdL[0]', LdL[0])

                if np.abs(LdL[0]) - np.abs(old_error) > 0:
                    print('#################################### WRONG | GRADIENT INCREASING')
                old_error = LdL[0]

            rep = rep + 1


if __name__ == "__main__":

    # You can set up a seed here for reproducibility
    rng = np.random.default_rng()

    kinetic_ising = ising(netsize=10, rng=rng)
    kinetic_ising.random_fields()
    kinetic_ising.random_wiring()
    hidden_ising = HiddenIsing(kinetic_ising, visible_units_per=0.1, b_size=1, rng=rng)
    hidden_ising.random_wiring()

    hidden_ising.sim_fit()
