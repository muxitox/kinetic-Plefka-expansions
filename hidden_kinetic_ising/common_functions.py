import numpy as np
import matplotlib.pyplot as plt


def simulate_full(ising, visible_idx, T, burn_in=0):
    """
    Simulate the full Kinetic Ising model to produce data

    :param T: number of steps to simulate
    :param burn_in:
    :return:
    """
    full_s = []
    s = []
    for t in range(0, T + burn_in):
        ising.ParallelUpdate()
        full_s.append(ising.s)
        if t >= burn_in:
            s.append(ising.s[visible_idx])
    # print('Spins', s)
    return full_s, s

def averaged_MSE(hidden_ising, moments, T_sim, num_simulations, burn_in=0):
    m, C, D = moments
    MSE_m = 0
    MSE_C = 0
    MSE_D = 0

    # Repeat the simulations to have a good estimation of the error
    for i in range(0, num_simulations):
        sim_s = hidden_ising.simulate_hidden(T_sim, burn_in=burn_in)
        sim_m, sim_C, sim_D = hidden_ising.compute_moments(sim_s, T_sim)

        MSE_m += np.mean((m - sim_m) ** 2)
        MSE_C += np.mean((C - sim_C) ** 2)
        MSE_D += np.mean((D - sim_D) ** 2)

    MSE_m /= num_simulations
    MSE_C /= num_simulations
    MSE_D /= num_simulations

    return MSE_m, MSE_C, MSE_D

def plot_likelihood_MSE(ell_list, error_list, eta, error_iter_list, MSE_m_list, MSE_C_list, MSE_D_list,
                             title_str):

    fig, ax = plt.subplots(2, figsize=(16, 10), dpi=100)
    ax[0].plot(ell_list, label='log(ell)')
    ax[0].plot(np.square(error_list), label='max_grad^2')
    ax[0].plot(np.diff(ell_list / eta), '--', label='np.diff(log_ell)/eta')
    ax[0].set_xlabel('iters')
    ax[0].legend()

    ax[1].plot(error_iter_list, MSE_m_list, label='MSE m')
    ax[1].plot(error_iter_list, MSE_C_list, label='MSE C')
    ax[1].plot(error_iter_list, MSE_D_list, label='MSE D')
    ax[1].set_xlabel('iters')
    ax[1].legend()

    fig.suptitle(title_str)

    return fig, ax

