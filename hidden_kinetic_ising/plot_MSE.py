import matplotlib.pyplot as plt
import os
import numpy as np

fig, ax = plt.subplots(3, figsize=(16, 10), dpi=100)

# Set the directory you want to start from
rootDir = '.'
for dirName, subdirList, fileList in os.walk(rootDir):
    head, num_vis = os.path.split(dirName)
    head, num_neurons = os.path.split(head)
    if str.isdigit(num_neurons) and str.isdigit(num_vis):

        fileList = sorted(fileList, key=lambda x: int(x.split('_')[3]))

        MSE_m_list = []
        MSE_D_list = []
        MSE_C_list = []
        hidden_list = []
        for fname in fileList:
            name, ext = os.path.splitext(fname)
            if ext == '.npz':
                split_name = name.split('_')
                num_hidden = split_name[3]
                data = np.load(os.path.join(dirName, fname))
                MSE_m = data['MSE_m']
                MSE_C = data['MSE_C']
                MSE_D = data['MSE_D']

                MSE_m_list.append(MSE_m)
                MSE_C_list.append(MSE_C)
                MSE_D_list.append(MSE_D)
                hidden_list.append(num_hidden)

        ax[0].plot(hidden_list, MSE_m_list, label=f'Original netsize: {num_neurons}. Visible units size: {num_vis}')
        ax[1].plot(hidden_list, MSE_C_list, label=f'Original netsize: {num_neurons}. Visible units size: {num_vis}')
        ax[2].plot(hidden_list, MSE_D_list, label=f'Original netsize: {num_neurons}. Visible units size: {num_vis}')

ax[0].set_ylabel('MSE m')
ax[1].set_ylabel('MSE C')
ax[2].set_ylabel('MSE D')
ax[2].set_xlabel('Number of hidden units')


plt.legend()

plt.savefig('results/MSE')



