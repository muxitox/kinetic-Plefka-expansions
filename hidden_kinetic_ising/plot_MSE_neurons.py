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


version_folder = 'it2'
# Set the directory you want to start from
rootDir = f'./results/neurons/{version_folder}/'
for dirName, subdirList, fileList in os.walk(rootDir):
    head, num_vis = os.path.split(dirName)
    _, dataset = os.path.split(head)
    if str.isdigit(dataset) and str.isdigit(num_vis):

        MSE_m_list = []
        MSE_D_list = []
        MSE_C_list = []
        log_ell_list = []
        hidden_list = []

        # Get it3 results
        it3_path = f'./results/neurons/it3/{dataset}/{num_vis}'
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

        hidden_code_pos = 2
        new_fileList = sorted(new_fileList, key=lambda x: int(x.split('_')[hidden_code_pos]))

        for fname in new_fileList:
                split_name = fname.split('_')

                num_hidden = split_name[hidden_code_pos]
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
                   label=f'Dataset: {dataset}. Visible units size: {num_vis}')
        ax[1].plot(hidden_list, MSE_C_list, color=colors[i],
                   label=f'Dataset: {dataset}. Visible units size: {num_vis}')
        ax[2].plot(hidden_list, MSE_D_list, color=colors[i],
                   label=f'Dataset: {dataset}. Visible units size: {num_vis}')
        ax[3].plot(hidden_list, log_ell_list, color=colors[i],
                   label=f'Dataset: {dataset}. Visible units size: {num_vis}')

        i += 1


ax[0].set_ylabel('MSE m')
ax[1].set_ylabel('MSE C')
ax[2].set_ylabel('MSE D')
ax[3].set_ylabel('Likelihood')

ax[3].set_xlabel('Number of hidden units')


plt.legend()

# plt.show()
plt.savefig(f'results/MSE/MSE_neurons_{dataset}')



