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

# Load forward

base_path = 'data/angel/forward/'
files = [f for f in listdir(base_path) if isfile(join(base_path, f))]

files = sorted(files, key = lambda x: int(x.split('_')[1]))
betas_s = [float(f.split('_')[1])/100 for f in files]

# LOAD DATA
mP_CMS_mean = np.zeros(len(files))
mP_TAP_mean = np.zeros(len(files))
m_MS_mean = np.zeros(len(files))
m_gt_mean = np.zeros(len(files))

CP_CMS_mean = np.zeros(len(files))
CP_TAP_mean = np.zeros(len(files))
C_MS_mean = np.zeros(len(files))
C_gt_mean = np.zeros(len(files))

DP_CMS_mean = np.zeros(len(files))
DP_TAP_mean = np.zeros(len(files))
D_MS_mean = np.zeros(len(files))
D_gt_mean = np.zeros(len(files))

for i, f in enumerate(files):
    data = np.load(join(base_path, f))
    mP_TAP_mean[i] = np.mean(data['mP_t1_t_mean'])
    m_MS_mean[i] = np.mean(data['mP_t1_mean'])
    mP_CMS_mean[i] = np.mean(data['mP_CMS_mean'])
    m_gt_mean[i] = np.mean(data['mPexp_mean'])

    CP_TAP_mean[i] = np.mean(data['CP_t1_t_mean'])
    C_MS_mean[i] = np.mean(data['CP_t1_mean'])
    CP_CMS_mean[i] = np.mean(data['CP_CMS_mean'])
    C_gt_mean[i] = np.mean(data['CPexp_mean'])

    DP_TAP_mean[i] = np.mean(data['DP_t1_t_mean'])
    D_MS_mean[i] = np.mean(data['DP_t1_mean'])
    DP_CMS_mean[i] = np.mean(data['DP_CMS_mean'])
    D_gt_mean[i] = np.mean(data['DPexp_mean'])


# Load reconstruction

base_path = 'data/angel/reconstruction/'
files = [f for f in listdir(base_path) if isfile(join(base_path, f))]
files = [f for f in files if '_r_' in f]

files = sorted(files, key=lambda x: int(x.split('_')[2]))
betas = [float(f.split('_')[2])/100 for f in files]
B = 201
betas = 1 + np.linspace(-1, 1, B) * 0.3

mP_CMS_r_mean = np.zeros(len(files))
mP_TAP_r_mean = np.zeros(len(files))
m_MS_r_mean = np.zeros(len(files))

CP_CMS_r_mean = np.zeros(len(files))
CP_TAP_r_mean = np.zeros(len(files))
C_MS_r_mean = np.zeros(len(files))

DP_CMS_r_mean = np.zeros(len(files))
DP_TAP_r_mean = np.zeros(len(files))
D_MS_r_mean = np.zeros(len(files))

for i, f in enumerate(files):
    data = np.load(join(base_path, f))
    mP_TAP_r_mean[i] = np.mean(data['mP_t1_t_mean'])
    m_MS_r_mean[i] = np.mean(data['mP_t1_mean'])
    mP_CMS_r_mean[i] = np.mean(data['mP_CMS_mean'])

    CP_TAP_r_mean[i] = np.mean(data['CP_t1_t_mean'])
    C_MS_r_mean[i] = np.mean(data['CP_t1_mean'])
    CP_CMS_r_mean[i] = np.mean(data['CP_CMS_mean'])

    DP_TAP_r_mean[i] = np.mean(data['DP_t1_t_mean'])
    D_MS_r_mean[i] = np.mean(data['DP_t1_mean'])
    DP_CMS_r_mean[i] = np.mean(data['DP_CMS_mean'])



# PLOT Forward
fig, ax = plt.subplots(2, 3)

ax[0][0].plot(betas_s, mP_TAP_mean, dashes=line[0],
               color=colors[0], label=labels[0])
ax[0][0].plot(betas_s, m_MS_mean, dashes=line[1],
           color=colors[1], label=labels[1])
ax[0][0].plot(betas_s, mP_CMS_mean, dashes=line[2],
           color=colors[2], label=labels[2])
ax[0][0].plot(betas_s, m_gt_mean,'k', dashes=line[3], lw=0.8)  # , label=r'$P$')
ax[0][0].set_xlabel('B')
ax[0][0].set_ylabel('<m>')
ax[0][0].axvline(x=1.0, color='k', lw=0.5)
ax[0][0].legend()


ax[0][1].plot(betas_s, CP_TAP_mean, dashes=line[0],
               color=colors[0], label=labels[0])
ax[0][1].plot(betas_s, C_MS_mean, dashes=line[1],
           color=colors[1], label=labels[1])
ax[0][1].plot(betas_s, CP_CMS_mean, dashes=line[2],
           color=colors[2], label=labels[2])
ax[0][1].plot(betas_s, C_gt_mean,'k', dashes=line[3], lw=0.8)  # , label=r'$P$')
ax[0][1].set_xlabel('B')
ax[0][1].set_ylabel('<C>')
ax[0][1].axvline(x=1.0, color='k', lw=0.5)
ax[0][1].legend()

ax[0][2].plot(betas_s, DP_TAP_mean, dashes=line[0],
               color=colors[0], label=labels[0])
ax[0][2].plot(betas_s, D_MS_mean, dashes=line[1],
           color=colors[1], label=labels[1])
ax[0][2].plot(betas_s, DP_CMS_mean, dashes=line[2],
           color=colors[2], label=labels[2])
ax[0][2].plot(betas_s, D_gt_mean,'k', dashes=line[3], lw=0.8)  # , label=r'$P$')
ax[0][2].set_xlabel('B')
ax[0][2].set_ylabel('<D>')
ax[0][2].axvline(x=1.0, color='k', lw=0.5)
ax[0][2].legend()

# Plot reconstruction

ax[1][0].plot(betas, mP_TAP_r_mean, dashes=line[0],
               color=colors[0], label=labels[0])
ax[1][0].plot(betas, m_MS_r_mean, dashes=line[1],
           color=colors[1], label=labels[1])
ax[1][0].plot(betas, mP_CMS_r_mean, dashes=line[2],
           color=colors[2], label=labels[2])
ax[1][0].plot(betas_s, m_gt_mean,'k', dashes=line[3], lw=0.8)  # , label=r'$P$')
ax[1][0].set_xlabel('B')
ax[1][0].set_ylabel('<m>')
ax[1][0].axvline(x=1.0, color='k', lw=0.5)
ax[1][0].legend()


ax[1][1].plot(betas, CP_TAP_r_mean, dashes=line[0],
               color=colors[0], label=labels[0])
ax[1][1].plot(betas, C_MS_r_mean, dashes=line[1],
           color=colors[1], label=labels[1])
ax[1][1].plot(betas, CP_CMS_r_mean, dashes=line[2],
           color=colors[2], label=labels[2])
ax[1][1].plot(betas_s, C_gt_mean,'k', dashes=line[3], lw=0.8)  # , label=r'$P$')
ax[1][1].set_xlabel('B')
ax[1][1].set_ylabel('<C>')
ax[1][1].axvline(x=1.0, color='k', lw=0.5)
ax[1][1].legend()

ax[1][2].plot(betas, DP_TAP_r_mean, dashes=line[0],
               color=colors[0], label=labels[0])
ax[1][2].plot(betas, D_MS_r_mean, dashes=line[1],
           color=colors[1], label=labels[1])
ax[1][2].plot(betas, DP_CMS_r_mean, dashes=line[2],
           color=colors[2], label=labels[2])
ax[1][2].plot(betas_s, D_gt_mean,'k', dashes=line[3], lw=0.8)  # , label=r'$P$')
ax[1][2].set_xlabel('B')
ax[1][2].set_ylabel('<D>')
ax[1][2].axvline(x=1.0, color='k', lw=0.5)
ax[1][2].legend()

plt.show()