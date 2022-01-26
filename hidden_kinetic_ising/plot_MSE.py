import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib import cm

fig, ax = plt.subplots(3, figsize=(16, 10), dpi=100)


colors = []
cmap = cm.get_cmap('plasma_r')
for i in range(0, 6):
    colors += [cmap((i) / 4)]

i = 0

# Set the directory you want to start from
rootDir = './results/stronger_couplings'
for dirName, subdirList, fileList in os.walk(rootDir):
    head, num_vis = os.path.split(dirName)
    head, num_neurons = os.path.split(head)
    if str.isdigit(num_neurons) and str.isdigit(num_vis):



        MSE_m_list = []
        MSE_D_list = []
        MSE_C_list = []
        hidden_list = []
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

                MSE_m_list.append(MSE_m)
                MSE_C_list.append(MSE_C)
                MSE_D_list.append(MSE_D)
                hidden_list.append(num_hidden)

        ax[0].plot(hidden_list, MSE_m_list, color=colors[i], label=f'Original netsize: {num_neurons}. Visible units size: {num_vis}')
        ax[1].plot(hidden_list, MSE_C_list, color=colors[i], label=f'Original netsize: {num_neurons}. Visible units size: {num_vis}')
        ax[2].plot(hidden_list, MSE_D_list, color=colors[i], label=f'Original netsize: {num_neurons}. Visible units size: {num_vis}')

        i += 1

ax[0].set_ylabel('MSE m')
ax[1].set_ylabel('MSE C')
ax[2].set_ylabel('MSE D')
ax[2].set_xlabel('Number of hidden units')


plt.legend()

# plt.show()
plt.savefig('results/stronger_couplings/MSE')



