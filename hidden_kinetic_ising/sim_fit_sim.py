from hidden_kinetic_ising_it2 import HiddenIsing
import numpy as np
from kinetic_ising import ising
import os
import matplotlib.pyplot as plt



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

# Important parameters for learning
original_netsize = 10
vis_units = 6


kinetic_ising = ising(netsize=original_netsize, rng=rng)
kinetic_ising.random_fields()
kinetic_ising.random_wiring()

hidden_ising = HiddenIsing(kinetic_ising, visible_size=vis_units, rng=rng)
T_ori = 500
burn_in = 100
full_s, visible_s = hidden_ising.simulate_full(T_ori, burn_in=burn_in)

m, C, D = hidden_ising.compute_moments(visible_s, T_ori)

# Make a second full simulation to have a baseline of the error due to the stochastic process
_, visible_s1 = hidden_ising.simulate_full(T_ori, burn_in=burn_in)
m1, C1, D1 = hidden_ising.compute_moments(visible_s1, T_ori)

print('Comparison between full models to observe the error in the simulation')
# print('m', m)
# print('m1', m1)
# print()
# print('C', C)
# print('C1', C1)
# print()
# print('D', D)
# print('D1', D1)

MSE_m = np.mean((m - m1) ** 2)
MSE_C = np.mean((C - C1) ** 2)
MSE_D = np.mean((D - D1) ** 2)

print('MSE m', MSE_m, 'C', MSE_C, 'D', MSE_D)

b_units_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
T_sim = 2000
eta = 0.01
max_reps = 6500
original_moments = (m, C, D)
for b_size in b_units_list:

    hidden_ising.set_hidden_size(b_size=b_size)
    hidden_ising.random_wiring()


    gradient_mode = 'regular'
    num_reps, ell_list, error_list, MSE_m_list, MSE_C_list, MSE_D_list, error_iter_list = \
        hidden_ising.fit(visible_s, eta, max_reps, T_ori, T_sim, original_moments, gradient_mode=gradient_mode)

    title_str = f'Seed: {seed}. Original size: {original_netsize}. Visible units: {vis_units}. Hidden units: {b_size}.' \
                f' O. Simulation steps: {T_ori}. F. Simulation steps: {T_sim}. eta: {eta}. max_reps: {max_reps} '
    print(title_str)

    num_simulations = 5

    f_MSE_m = 0
    f_MSE_C = 0
    f_MSE_D = 0

    # Repeat the simulations to have a good estimation of the error
    for i in range(0, num_simulations):
        sim_s = hidden_ising.simulate_hidden(T_sim, burn_in=burn_in)
        f_sim_m, f_sim_C, f_sim_D = hidden_ising.compute_moments(sim_s, T_sim)

        f_MSE_m += np.mean((m - f_sim_m) ** 2)
        f_MSE_C += np.mean((C - f_sim_C) ** 2)
        f_MSE_D += np.mean((D - f_sim_D) ** 2)

    f_MSE_m /= num_simulations
    f_MSE_C /= num_simulations
    f_MSE_D /= num_simulations

    MSE_m_list.append(f_MSE_m)
    MSE_C_list.append(f_MSE_C)
    MSE_D_list.append(f_MSE_D)
    error_iter_list.append(max_reps)

    print('Final MSE m', f_MSE_m, 'C', f_MSE_C, 'D', f_MSE_D)

    print()


    fig, ax = plt.subplots(2, figsize=(16, 10), dpi=100)
    ax[0].plot(ell_list[0:num_reps], label='log(ell)')
    ax[0].plot(np.square(error_list[0:num_reps]), label='max_grad^2')
    ax[0].plot(np.diff(ell_list[0:num_reps] / eta), '--', label='np.diff(log_ell)/eta')
    ax[0].set_xlabel('iters')
    ax[0].legend()

    ax[1].plot(error_iter_list, MSE_m_list, label='MSE m')
    ax[1].plot(error_iter_list, MSE_C_list, label='MSE C')
    ax[1].plot(error_iter_list, MSE_D_list, label='MSE D')
    ax[1].set_xlabel('iters')
    ax[1].legend()

    fig.suptitle(title_str)

    path = f'results/size_plus_sqrt_size/{original_netsize}/{vis_units}/'

    # Check whether the specified path exists or not
    isExist = os.path.exists(path)

    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(path)
        print(f"The new directory \"{path} \" is created!")

    eta_str = str(eta).replace('.', '')
    filename = f"{seed}_{original_netsize}_{vis_units}_{b_size}_{T_ori}_{T_sim}_eta{eta_str}_{max_reps}_{burn_in}"
    plt.savefig(path + filename)

    np.savez_compressed(path + filename+'.npz',
                        H=hidden_ising.H,
                        J=hidden_ising.J,
                        M=hidden_ising.M,
                        K=hidden_ising.K,
                        L=hidden_ising.L,
                        b0=hidden_ising.b_0,
                        m=m,
                        C=C,
                        D=D,
                        MSE_m=f_MSE_m,
                        MSE_C=f_MSE_C,
                        MSE_D=f_MSE_D)

print('Seed', seed)