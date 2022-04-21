import numpy as np
from kinetic_ising import ising
import os
import matplotlib.pyplot as plt

from hidden_kinetic_ising_it2 import HiddenIsing as HI2
from utils.common_functions import simulate_full, plot_likelihood_MSE, compute_moments


# You can set up a seed here for reproducibility
# Seed to check wrong behavior: 6, 2425, 615
# 3656, 0.6 1, 2, 3

# 3813, 4075, 2692, 3262, 3988
reproducible = True
if reproducible:
    seed = [1185]
else:
    seed = np.random.randint(5000, size=1)

rng = np.random.default_rng(seed)

print('Seed', seed)

# Important parameters for learning
original_netsize = 10
vis_units = 6
max_reps = 4000
gradient_mode = 'regular'
folder_code = 'size'
save_results = True

kinetic_ising = ising(netsize=original_netsize, rng=rng)
kinetic_ising.random_fields()
kinetic_ising.random_wiring()
hidden_ising = HI2(visible_size=vis_units, rng=rng)
T_ori = 1500
T_sim = 1500
burn_in = 100

visible_idx = rng.choice(range(0, kinetic_ising.size), vis_units)
full_s = simulate_full(kinetic_ising, T_ori, burn_in=burn_in)
visible_s = full_s[:, visible_idx]

m, C, D = compute_moments(visible_s.T)

# Make a second full simulation to have a baseline of the error due to the stochastic process
full_s2 = simulate_full(kinetic_ising, T_ori, burn_in=burn_in)
visible_s2 = full_s2[:, visible_idx]
m2, C2, D2 = compute_moments(visible_s2.T)

print('Comparison between full models to observe the error in the simulation')

MSE_m = np.mean((m - m2) ** 2)
MSE_C = np.mean((C - C2) ** 2)
MSE_D = np.mean((D - D2) ** 2)

print('MSE m', MSE_m, 'C', MSE_C, 'D', MSE_D)

# b_units_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
b_units_list = [0, 1]
eta = 0.01
original_moments = (m, C, D)
for b_size in b_units_list:

    # seed = 'copy'
    # bg = np.load('../neurons/bg.npy', allow_pickle=True).item()
    #
    # rng = np.random.default_rng(bg)

    hidden_ising.set_hidden_size(b_size=b_size)
    hidden_ising.rng = np.random.default_rng(seed)
    hidden_ising.random_wiring()

    ell_list, error_list, MSE_m_list, MSE_C_list, MSE_D_list, error_iter_list, MSE_model_m, MSE_model_D, J_mean_list, J_var_list = \
        hidden_ising.fit(visible_s, eta, max_reps, T_ori, T_sim, original_moments, gradient_mode=gradient_mode)

    title_str = f'Seed: {seed}. Original size: {original_netsize}. Visible units: {vis_units}. Hidden units: {b_size}.' \
                f' O. Simulation steps: {T_ori}. F. Simulation steps: {T_sim}. eta: {eta}. max_reps: {max_reps} '
    print(title_str)

    print('Final MSE m', MSE_m_list[-1], 'C', MSE_C_list[-1], 'D', MSE_D_list[-1])
    print('Final Log-Likelihood', ell_list[-1])
    print()

    # Configure the plot
    fig, ax = plot_likelihood_MSE(ell_list, error_list, eta, error_iter_list, MSE_m_list, MSE_C_list, MSE_D_list,
                                  title_str, MSE_model_m, MSE_model_D, J_mean_list, J_var_list)

    if save_results:

        path = f'results2/it2/{folder_code}/{original_netsize}/{vis_units}/{seed}/'

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
                            MSE_m=MSE_m_list[-1],
                            MSE_C=MSE_C_list[-1],
                            MSE_D=MSE_D_list[-1],
                            log_ell=ell_list[-1])
    else:
        plt.show()

print('Seed', seed)