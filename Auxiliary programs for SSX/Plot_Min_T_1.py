'''My own version of plotting tools to plot 2d slices around global minimum of T'''
import sys
import pathlib
from matplotlib import cm
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
T = data['tasks/T'][:,:,:,:]
#print(np.size(sim_time))

print(np.shape(sim_time))
t0 = 62
x0,y0,z0 = np.unravel_index(T[t0,:,:,:].argmin(),T[t0,:,:,:].shape)
print(T[t0,x0,y0,z0])
print('at t = ',sim_time[t0])
print('at x = ',x[x0])
print('at y = ',y[y0])
print('at z = ',z[z0])
Txy = []
Tyz = []
Txz = []
for i in range(np.size(x)):
    for j in range(np.size(y)):
        temp = []
        temp.append(T[t0,i,j,z0])
    Txy.append(temp)
    temp = []

for i in range(np.size(x)):
    for k in range(np.size(z)):
        temp = []
        temp.append(T[t0,i,y0,k])
    Txz.append(temp)
    temp = []

print(np.size(z))
for j in range(np.size(y)):
    for k in range(np.size(z)):
        temp = []
        temp.append(T[t0,x0,j,k])
    Tyz.append(temp)
    temp = []
Txy = np.array(Txy)
Txz = np.array(Txz)
Tyz = np.array(Tyz)

fig = plt.figure(figsize=plt.figaspect(0.33))

#1st
ax = fig.add_subplot(1, 3, 1, projection='3d')
y1, x1 = np.meshgrid(y, x)
surf = ax.plot_surface(x1, y1, Txy, rstride=1, cstride=1,cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('T')
#2nd
ax = fig.add_subplot(1, 3, 2, projection='3d')
z2,x2 = np.meshgrid(z,x)
surf = ax.plot_surface(x2, z2, Txz, rstride=1, cstride=1,cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
ax.set_xlabel('x')
ax.set_ylabel('z')
ax.set_zlabel('T')
#3rd
ax = fig.add_subplot(1, 3, 3, projection='3d')
z3,y3 = np.meshgrid(z,y)
print(np.shape(z3))
print(np.shape(y3))
print(np.shape(Tyz))
surf = ax.plot_surface(y3, z3, Tyz, rstride=1, cstride=1,cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
ax.set_xlabel('y')
ax.set_ylabel('z')
ax.set_zlabel('T')
plt.show()
plt.savefig('minimum_T.png')

'''
for t in range(np.size(sim_time)):
    #T^-3/2 for spitzer, 0.001 for regular
    max_v.append(np.max(np.sqrt(vx[t, :, :, :] ** 2 + vy[t, :, :, :] ** 2 + vz[t, :, :, :] ** 2)))
    resistivity.append(np.max(1/(np.sqrt(T[t,:, :, :])**3)))
    max_B.append(np.max(np.sqrt(Bx[t, :, :, :] ** 2 + By[t, :, :, :] ** 2 + Bz[t, :, :, :] ** 2)))
    max_J.append(np.max(np.sqrt(jx[t, :, :, :] ** 2 + jy[t, :, :, :] ** 2 + jz[t, :, :, :] ** 2)))

fig, axs = plt.subplots(2, 2)
axs[0, 0].plot(sim_time, max_v)
axs[0, 0].set_title("max_v")
axs[1, 0].plot(sim_time, max_J)
axs[1, 0].set_title("max J")
axs[1, 0].sharex(axs[0, 0])
axs[0, 1].plot(sim_time, max_B)
axs[0, 1].set_title("max_B")
axs[1, 1].plot(sim_time, resistivity)
axs[1, 1].set_title("resistivity")
fig.tight_layout()
plt.savefig('max v J B Resistivity')
'''
