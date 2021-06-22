import sys
import pathlib
import h5py
import os
import matplotlib.pyplot as plt

basedir = os.path.dirname(os.path.realpath(__file__))
dfile = basedir + '/integrals.h5'

data = h5py.File(str(dfile), "r")

z = data['scales/z/1'][:]
vx = data['tasks/vx_x'][-1,0,0,:]

plt.semilogy(z, vx)
plt.xlabel("y")
plt.ylabel(r"$<v_x>$")

plt.savefig('vx_yprof.png')