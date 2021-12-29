#!/usr/bin/env python3
"""

This code allows to run simulations of the kinetic Ising model,
with asymmetric weights and parallel updates.
"""

import numpy as np
from kinetic_ising import ising
import copy


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

        self.visible_size = int(self.size * visible_units_per)  # Network size
        self.hidden_size = self.size - self.visible_size
        self.b_size = b_size

        # self.H = np.zeros(self.visible_size)  # Fields
        self.J = np.zeros((self.visible_size, self.visible_size))  # Spin-to-Spin couplings
        self.M = np.zeros((self.visible_size, self.b_size))  # Hidden-to-Hidden couplings
        self.K = np.zeros((self.b_size, self.visible_size))  # Hidden-to-Neuron couplings
        self.L = np.zeros((self.b_size, self.b_size))  # Hidden-to-Hidden couplings
        self.b_0 = np.zeros(self.b_size)

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
        T = 8
        # Simulate the full Kinetic Ising model to produce data
        full_s = []
        s = []
        for t in range(0, T):
            self.ising.ParallelUpdate()
            full_s.append(self.ising.s)
            s.append(self.ising.s[self.visible_idx])

        # print('Spins', s)

        # Initialize variables for learning
        eta = 0.00001
        eta_b0 = 0.00001
        rep = 0
        max_reps = 4500
        error_lim = 0.001
        error = np.inf
        # Learning loop
        old_error_L = np.inf
        old_error_b0 = np.inf
        old_error_K = np.inf
        old_error_M = np.inf
        old_error_J = np.inf

        old_error = np.inf

        old_logl = -np.inf

        while error > error_lim and rep < max_reps:
            print(rep)

            # Initialize the gradients to 0
            dLdJ = np.zeros((self.visible_size, self.visible_size))
            dLdM = np.zeros((self.visible_size, self.b_size))
            dLdK = np.zeros((self.b_size, self.visible_size))
            dLdL = np.zeros((self.b_size, self.b_size))
            dLdb_0 = np.zeros(self.b_size)

            # Likelihood accumulator
            logl = 0

            # State of b neurons at time [t-1]
            b_t1 = self.b_0

            # In t==1 we need the derivative of b wrt K and L at t-1,
            # and that'd require s_{t-2} which does not exist at that time step
            b_t1_dK = np.zeros((self.b_size, self.b_size, self.visible_size))
            b_t1_dL = np.zeros((self.b_size, self.b_size, self.b_size))

            b_dK = np.zeros((self.b_size, self.b_size, self.visible_size))
            b_dL = np.zeros((self.b_size, self.b_size, self.b_size))

            # We start in index 1 because we do not have s_{t-1} for t=0
            for t in range(1, T):

                # Compute the derivative of the Likelihood wrt J
                h = np.dot(self.M, b_t1) + np.dot(self.J, s[t - 1])
                tanh_h = np.tanh(h)
                # print('h', np.dot(self.M, b_t1) + np.dot(self.J, s[t - 1]))
                # print('tanh_h', tanh_h)
                sub_s_tanhh = s[t] - tanh_h
                # print('sub_s_tanhh', sub_s_tanhh)

                # Compute the log Likelihood to check
                logl += np.dot(s[t], h) - np.sum(np.log(2 * np.cosh(h)))

                # Derivative of the Likelihood wrt J
                dLdJ += np.einsum('i,j->ij', sub_s_tanhh, s[t - 1])

                # Save computational load if the number of b neurons < 1
                if self.b_size > 0:
                    dLdM += np.einsum('i,j->ij', sub_s_tanhh, b_t1)

                    if t == 1:
                        # Compute the gradient of the Likelihood wrt b(0)
                        dLdb_0 = np.dot(sub_s_tanhh, self.M)

                    # Compute the Jacobians wrt K and L
                    for i in range(0, self.visible_size):
                        # Derivative of the Likelihood wrt K
                        dLdK += sub_s_tanhh[i] * np.einsum('g,gnm->nm', self.M[i, :], b_t1_dK)

                        # Derivative of the Likelihood wrt L
                        dLdL += sub_s_tanhh[i] * np.einsum('g,gnm->nm', self.M[i, :], b_t1_dL)


                    # Compute the necessary information for the next step
                    # At t==1, b(t-1)=0
                    b = np.tanh(np.dot(self.K, s[t - 1]) + np.dot(self.L, b_t1))
                    # print('b', b)

                    # Compute the derivatives of b wrt L and K
                    # At t==1 b_t1_dK=0 and b_t1_dL=0
                    for i in range(0, self.b_size):
                        # Derivative of b wrt K
                        b_dK[i] = np.einsum('g,gnm->nm', self.L[i, :], b_t1_dK)
                        b_dK[i, i, :] += s[t - 1]
                        b_dK[i] *= (1 - b[i] ** 2)

                        # Derivative of b wrt L
                        b_dL[i] = np.einsum('g,gnm->nm', self.L[i, :], b_t1_dL)
                        b_dL[i, i, :] += b_t1
                        b_dL[i] *= (1 - b[i] ** 2)

                    # Save the variables for the next step
                    b_t1 = copy.deepcopy(b)
                    b_t1_dK = copy.deepcopy(b_dK)
                    b_t1_dL = copy.deepcopy(b_dL)

            # Normalize the gradients temporally and by the number of spins in the sum of the Likelihood
            dLdJ /= T - 1
            dLdM /= T - 1
            dLdK /= self.visible_size * (T - 1)
            dLdL /= self.visible_size * (T - 1)
            dLdb_0 /= self.visible_size

            # self.J = self.J + eta * dLdJ
            # self.K = self.K + eta * LdK
            # self.L = self.L + eta * LdL
            self.M = self.M + eta * dLdM
            if rep < 100:
                self.b_0 = self.b_0 + eta_b0 * dLdb_0

            # print('b0', self.b_0)
            # print('L', self.L)

            # Prints for debugging
            # if self.b_size > 1:
            #     self.L[1][1] = self.L[1][1] + eta * LdL[1][1]
            # else:
            #     self.L[0] = self.L[0] + eta * LdL[0]

            # print('J', self.J)
            print('M', self.M)
            print('b0', self.b_0)
            # print('K', self.K)
            # print('L', self.L)
            # print()

            # Compute the error as the max component of the gradients
            # if self.b_size > 1:
            #     error = max(np.max(np.abs(dLdJ)), np.max(np.abs(LdM)), np.max(np.abs(LdK)), np.max(np.abs(LdL)))
            #
            #     if np.abs(error) > np.abs(old_error):
            #         print('#################################### WRONG | GRADIENT INCREASING')
            #
            #     old_error = error
            # else:
            #     error = np.max(np.abs(dLdJ))
            #
            #     if np.abs(error) > np.abs(old_error):
            #         print('#################################### WRONG | GRADIENT INCREASING')
            #
            #     old_error = error
            #
            # print(error)

            print('log Likelihood', logl)

            if old_logl > logl:
                print('#################################### WRONG | LIKELIHOOD DECREASING')

            # print('Comparison', '(logl-old_logl)/eta', (logl-old_logl)/eta, 'dLdM²', old_error_M**2)

            old_logl = logl


            if self.b_size > 1:
                print('LdL[1][1]', dLdL[1][1])

                if np.abs(dLdL[1][1]) - np.abs(old_error_L) > 0:
                    print('#################################### WRONG | L GRADIENT INCREASING')
                old_error_L = dLdL[1][1]

            else:
                # print('dLdL[0]', LdL[0])
                print('dLdM[0]', dLdM[0])

                # print('dLdJ[0]', dLdJ[0])
                # print('dLdK[0]', LdK[0])
                print('dLdb0', dLdb_0)

                # if np.abs(LdL[0]) - np.abs(old_error_L) > 0:
                #     print('#################################### WRONG | L GRADIENT INCREASING')

                if np.abs(dLdM[0]) - np.abs(old_error_M) > 0:
                    print('#################################### WRONG | M GRADIENT INCREASING')
                #
                # if np.abs(dLdJ[0]) - np.abs(old_error_J) > 0:
                #     print('#################################### WRONG | J GRADIENT INCREASING')
                #
                # if np.abs(LdK[0]) - np.abs(old_error_K) > 0:
                #     print('#################################### WRONG | K GRADIENT INCREASING')

                if np.abs(dLdb_0) - np.abs(old_error_b0) > 0:
                    print('#################################### WRONG | b GRADIENT INCREASING')
                old_error_L = dLdL[0]
                old_error_b0 = dLdb_0
                old_error_K = dLdK[0]
                old_error_M = dLdM[0]
                old_error_J = dLdJ[0]




            rep = rep + 1

            print()


if __name__ == "__main__":
    # You can set up a seed here for reproducibility
    # Seed to check wrong behavior: 6

    reproducible = True
    if reproducible:
        seed = 2425
    else:
        seed = np.random.randint(5000, size=1)

    rng = np.random.default_rng(seed)

    print('Seed', seed)


    kinetic_ising = ising(netsize=10, rng=rng)
    kinetic_ising.random_fields()
    kinetic_ising.random_wiring()
    hidden_ising = HiddenIsing(kinetic_ising, visible_units_per=0.3, b_size=2, rng=rng)
    hidden_ising.random_wiring()

    hidden_ising.sim_fit()

    print('Seed', seed)
