import numpy as np
import matplotlib.pyplot as plt


def simulate_full(ising, T, burn_in=0):
    """
    Simulate the full Kinetic Ising model to produce data

    :param ising:
    :param visible_idx:
    :param T: number of steps to simulate
    :param burn_in:
    :return:
    """

    full_s = np.zeros((T, ising.size))
    for t in range(0, T + burn_in):
        ising.ParallelUpdate()
        if t >= burn_in:
            full_s[t-burn_in] = ising.s
    # print('Spins', s)
    return full_s


def averaged_MSE(hidden_ising, gt_moments, T_sim, num_simulations, burn_in=0):
    m, C, D = gt_moments
    MSE_m = 0
    MSE_C = 0
    MSE_D = 0

    # Repeat the simulations to have a good estimation of the error
    for i in range(0, num_simulations):
        sim_s = hidden_ising.simulate_visible(T_sim, burn_in=burn_in)
        sim_m, sim_C, sim_D = compute_moments(sim_s.T)

        MSE_m += np.mean((m - sim_m) ** 2)
        MSE_C += np.mean((C - sim_C) ** 2)
        MSE_D += np.mean((D - sim_D) ** 2)

    MSE_m /= num_simulations
    MSE_C /= num_simulations
    MSE_D /= num_simulations

    return MSE_m, MSE_C, MSE_D

def compute_moments(X, d=1):
    """
    Computes the statistical moments m (mean), C (same-time correlations) and D (delayed correlations) of X
    :param X: data matrix of shape (N, T). N: number of neurons. T: time steps
    :param d:
    :return:
    """

    # X = X.T
    N, T = X.shape

    m = np.einsum('it->i', X) / T
    # m = X.mean(axis=1)


    C = np.einsum('it,jt->ij', X, X) / T
    D = np.einsum('it,jt->ij', X[:, :-d], X[:, d:]) / (T - 1)

    mi = np.einsum('it->i', X[:, :-d]) / (T - 1)
    mii = np.einsum('it->i', X[:, d:]) / (T - 1)

    C -= np.einsum('i,k->ik', m, m, optimize=True)
    D -= np.einsum('i,k->ik', mi, mii, optimize=True)
    C[range(N), range(N)] = 1 - m ** 2

    return m, C, D


def compute_moments_sparse(X, d=1):
    N, T = X.shape

    m, C, D = compute_moments(X, d)

    C *= 4
    D *= 4
    m = m * 2 - 1
    C[range(N), range(N)] = 1 - m ** 2
    return m, C, D


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
