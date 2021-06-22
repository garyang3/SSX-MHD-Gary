'''My own version of plotting tools to plot for the Rayleigh bernard problem'''
import sys
import pathlib
import h5py
import os
import matplotlib.pyplot as plt

basedir = os.path.dirname(os.path.realpath(__file__))
dfile = basedir + '/snapshots/snapshots.h5'

data = h5py.File(str(dfile), "r")

x = data['scales/x/4'][:]
y = data['scales/y/4'][:]
t = data['scales/sim_time'][:]
bint4 = data['tasks/b integral x4'][32,:,:,0]

plt.pcolormesh(x, y, bint4)
plt.xlabel("x")
plt.ylabel("y")
plt.title('t=%1.3f' %t[32])
plt.savefig('b_int vs x vs y. at.t=.png')
plt.show()
'''
plt.semilogy(z, vx)
plt.xlabel("y")
plt.ylabel(r"$<v_x>$")

plt.savefig('b_int vs x vs y.png')
'''