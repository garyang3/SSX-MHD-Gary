'''My own version of plotting tools to plot for the Rayleigh bernard problem'''
import sys
import pathlib
import h5py
import os
import numpy as np
import matplotlib.pyplot as plt

basedir = os.path.dirname(os.path.realpath(__file__))
dfile = basedir + '/fields_two_s1.h5'

data = h5py.File(str(dfile), "r")

x = data['scales/x/1.0'][:]
y = data['scales/y/1.0'][:]
z = data['scales/z/1.0'][:]
sim_time = data['scales/sim_time'][:]
vx = data['tasks/vx'][:,:,:,:]
vy = data['tasks/vy'][:,:,:,:]
vz = data['tasks/vz'][:,:,:,:]
Bx = data['tasks/Bx'][:,:,:,:]
By = data['tasks/By'][:,:,:,:]
Bz = data['tasks/Bz'][:,:,:,:]
jx = data['tasks/jx'][:,:,:,:]
jy = data['tasks/jy'][:,:,:,:]
jz = data['tasks/jz'][:,:,:,:]
T = data['tasks/T'][:,:,:,:]
#print(np.size(sim_time))
max_v = []
max_J = []
max_B = []
resistivity = []
max_T = []
for t in range(np.size(sim_time)):
    '''T^-3/2 for spitzer, 0.001 for regular'''
    max_v.append(np.max(np.sqrt(vx[t, :, :, :] ** 2 + vy[t, :, :, :] ** 2 + vz[t, :, :, :] ** 2)))
    #resistivity.append(np.max(1/(np.sqrt(T[t,:, :, :])**3)))
    max_T.append(np.max(T[t, :, :, :]))
    max_B.append(np.max(np.sqrt(Bx[t, :, :, :] ** 2 + By[t, :, :, :] ** 2 + Bz[t, :, :, :] ** 2)))
    max_J.append(np.max(np.sqrt(jx[t, :, :, :] ** 2 + jy[t, :, :, :] ** 2 + jz[t, :, :, :] ** 2)))

fig, axs = plt.subplots(2, 2)
axs[0, 0].plot(sim_time, max_v,marker='.')

axs[0, 0].set_title("max_v")
axs[1, 0].plot(sim_time, max_J,marker='.')

axs[1, 0].set_title("max J")
axs[1, 0].sharex(axs[0, 0])
axs[0, 1].plot(sim_time, max_B,marker='.')

axs[0, 1].set_title("max_B")
axs[1, 1].plot(sim_time, max_T,marker='.')
axs[1, 1].set_title("max T")
fig.tight_layout()
plt.savefig('max v J B T.png')
