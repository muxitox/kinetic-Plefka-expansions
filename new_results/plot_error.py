import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm

from os import listdir
from os.path import isfile, join

size = 512
R = 1000000

T = 2**7

labels = [
        r'TAP',
        r'MS',
        r'CMS',
       ]

cmap = cm.get_cmap('plasma_r')
colors = []
for i in range(4):
    colors += [cmap((i) / 3)]

line = [(5, 4), (5, 4), '', (5, 4)]



base_path = 'data/angel/forward/'
files = [f for f in listdir(base_path) if isfile(join(base_path, f))]

files = sorted(files, key = lambda x: int(x.split('_')[1]))
betas_s = [float(f.split('_')[1])/100 for f in files]

# LOAD DATA

EmP_TAP_mean = np.zeros(len(files))
Em_MS_mean = np.zeros(len(files))
EmP_CMS_mean = np.zeros(len(files))

ECP_TAP_mean = np.zeros(len(files))
EC_MS_mean = np.zeros(len(files))
ECP_CMS_mean = np.zeros(len(files))

EDP_TAP_mean =np.zeros(len(files))
ED_MS_mean = np.zeros(len(files))
EDP_CMS_mean = np.zeros(len(files))



for i, f in enumerate(files):
    data = np.load(join(base_path, f))
    EmP_TAP_mean[i] = np.mean(data['EmP_t1_t'])
    Em_MS_mean[i] = np.mean(data['EmP_t1'])
    EmP_CMS_mean[i] = np.mean(data['Em_CMS'])

    ECP_TAP_mean[i] = np.mean(data['ECP_t1_t'])
    EC_MS_mean[i] = np.mean(data['ECP_t1'])
    ECP_CMS_mean[i] = np.mean(data['ECP_CMS'])

    EDP_TAP_mean[i] = np.mean(data['EDP_t1_t'])
    ED_MS_mean[i] = np.mean(data['EDP_t1'])
    EDP_CMS_mean[i] = np.mean(data['EDP_CMS'])



# PLOT Forward
fig, ax = plt.subplots(1, 3)

ax[0].semilogy(betas_s, EmP_TAP_mean, dashes=line[0],
               color=colors[0], label=labels[0])
ax[0].semilogy(betas_s, Em_MS_mean, dashes=line[1],
           color=colors[1], label=labels[1])
ax[0].semilogy(betas_s, ECP_TAP_mean, dashes=line[2],
           color=colors[2], label=labels[2])
ax[0].set_xlabel('B')
ax[0].set_ylabel('LOG MSE <m>')
ax[0].axvline(x=1.0, color='k', lw=0.5)
ax[0].legend()


ax[1].semilogy(betas_s, ECP_TAP_mean, dashes=line[0],
               color=colors[0], label=labels[0])
ax[1].semilogy(betas_s, EC_MS_mean, dashes=line[1],
           color=colors[1], label=labels[1])
ax[1].semilogy(betas_s, ECP_CMS_mean, dashes=line[2],
           color=colors[2], label=labels[2])
ax[1].set_xlabel('B')
ax[1].set_ylabel('LOG MSE <C>')
ax[1].axvline(x=1.0, color='k', lw=0.5)
ax[1].legend()

ax[2].semilogy(betas_s, EDP_TAP_mean, dashes=line[0],
               color=colors[0], label=labels[0])
ax[2].semilogy(betas_s, ED_MS_mean, dashes=line[1],
           color=colors[1], label=labels[1])
ax[2].semilogy(betas_s, EDP_CMS_mean, dashes=line[2],
           color=colors[2], label=labels[2])
ax[2].set_xlabel('B')
ax[2].set_ylabel('LOG MSE <D>')
ax[2].axvline(x=1.0, color='k', lw=0.5)
ax[2].legend()



plt.show()