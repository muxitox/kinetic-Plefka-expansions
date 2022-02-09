import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib import cm

fig, ax = plt.subplots(4, figsize=(16, 10), dpi=100)


colors = []
cmap = cm.get_cmap('plasma_r')
for i in range(0, 5):
    colors += [cmap((i) / 4)]

i = 0

J_initialization = 'constantJ'


version_folder = 'it2'
# Set the directory you want to start from
rootDir = f'./results/{version_folder}/{J_initialization}'
for dirName, subdirList, fileList in os.walk(rootDir):
    head, num_vis = os.path.split(dirName)
    _, num_total_neurons = os.path.split(head)
    if str.isdigit(num_total_neurons) and str.isdigit(num_vis):

        MSE_m_list = []
        MSE_D_list = []
        MSE_C_list = []
        log_ell_list = []
        hidden_list = []

        # Get it3 results
        it3_path = f'./results/it3/{J_initialization}/{num_total_neurons}/{num_vis}'
        it3_filelist = [f for f in os.listdir(it3_path) if os.path.isfile(os.path.join(it3_path, f))]

        it3_filename = None
        for fname in it3_filelist:
            name, ext = os.path.splitext(fname)
            if ext == '.npz':
                it3_filename = fname

        data = np.load(os.path.join(it3_path, it3_filename))
        MSE_m = data['MSE_m']
        MSE_C = data['MSE_C']
        MSE_D = data['MSE_D']
        log_ell = data['log_ell']

        MSE_m_list.append(MSE_m)
        MSE_C_list.append(MSE_C)
        MSE_D_list.append(MSE_D)
        log_ell_list.append(np.exp(log_ell))
        hidden_list.append(-1)

        new_fileList = []
        for fname in fileList:
            name, ext = os.path.splitext(fname)
            if ext == '.npz':
                new_fileList.append(fname)

        new_fileList = sorted(new_fileList, key=lambda x: int(x.split('_')[3]))

        for fname in new_fileList:
                split_name = fname.split('_')

                num_hidden = split_name[3]
                data = np.load(os.path.join(dirName, fname))
                MSE_m = data['MSE_m']
                MSE_C = data['MSE_C']
                MSE_D = data['MSE_D']
                log_ell = data['log_ell']

                MSE_m_list.append(MSE_m)
                MSE_C_list.append(MSE_C)
                MSE_D_list.append(MSE_D)
                log_ell_list.append(np.exp(log_ell))
                hidden_list.append(num_hidden)

        ax[0].plot(hidden_list, MSE_m_list, color=colors[i],
                   label=f'Original netsize: {num_total_neurons}. Visible units size: {num_vis}')
        ax[1].plot(hidden_list, MSE_C_list, color=colors[i],
                   label=f'Original netsize: {num_total_neurons}. Visible units size: {num_vis}')
        ax[2].plot(hidden_list, MSE_D_list, color=colors[i],
                   label=f'Original netsize: {num_total_neurons}. Visible units size: {num_vis}')
        ax[3].plot(hidden_list, log_ell_list, color=colors[i],
                   label=f'Original netsize: {num_total_neurons}. Visible units size: {num_vis}')

        i += 1


ax[0].set_ylabel('MSE m')
ax[1].set_ylabel('MSE C')
ax[2].set_ylabel('MSE D')
ax[3].set_ylabel('Likelihood')

ax[3].set_xlabel('Number of hidden units')

# plt.suptitle('J = 1/self.size + self.rng.random((self.size, self.size)) / np.sqrt(self.size)')
# plt.suptitle('J = self.rng.random((self.size, self.size)) / np.sqrt(self.size)')
# plt.suptitle('J = self.rng.random((self.size, self.size)) / self.size')
plt.suptitle('J = 1/self.size')

plt.legend()

# plt.show()
plt.savefig(f'results/MSE/MSE_{J_initialization}')



