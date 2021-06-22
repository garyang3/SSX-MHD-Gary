"""
Last used/edited by Shouzhuo Yang, Swarthmore class of '23. Initially sent by Ercong Luo '21, Matiwos Mebratu '21 and Manjit (postdoc).
The docopt module did not combine well with my laptop, so I changed it a little bit that requires user to input arguments using sys.argv[], but it works essentially
the same. 

This version works for VAPOR 2.6.0., NOT for Vapor3. To download Vapor260, refer to this website below:
https://github.com/NCAR/vapor2website/releases?fbclid=IwAR0v6CIONuUxMLqMMIwyZ-1b-6n40eafqf6QfcxItU1xAI997rkqw7r-r_U
Converts 3D, time varying HDF5 data from Dedalus simulations to VAPOR vdf format
Usage:
    toVapor.py <fileIn> <fileOutName> [<dimensionRatio>]
If no netCDF file is created, you may have to call "source VAPORPATH" yourself. See below.
"""

import h5py
import os
import sys
import numpy as np
import netCDF4 as n4
from docopt import docopt
import matplotlib.pyplot as plt

'''Clearly, this only works on MacOS. To activate VAPOR environment on windows, see
https://www.vapor.ucar.edu/docs/vapor-installation/vapor-windows-binary-installation.
Moreover, the Dedalus path is specific to this Swarthmore Physics department laptop
"Manjit's MacBook Pro". Change these variables as needed.'''

PYTHONPATH = '/Applications/VAPOR/VAPOR.app/Contents/MacOS/python'
VAPORPATH = '/Applications/VAPOR/VAPOR.app/Contents/MacOS/vapor-setup.sh' # problem here


try:
    hf = h5py.File(sys.argv[1], 'r')
except IOError:
    sys.exit("Your input file could not be accessed")

try:
    tasks = list(hf['tasks'])
    shape = hf['tasks/%s' % tasks[0]].shape
except:
    sys.exit("There was an error reading the h5 data")

#get dimension sizes
nx = shape[1]
ny = shape[2]
nz = shape[3]
nt = shape[0]

#create dimension data
x = np.linspace(0,1,nx)
y = np.linspace(0,1,ny)
z = np.linspace(0,1,nz)
t = np.arange(float(nt))
print(shape)

#create new ncdf4 file
dataset = n4.Dataset(sys.argv[2]+'.nc', 'w', format='NETCDF4')

#create dimensions
xset = dataset.createDimension('x', nx)
yset = dataset.createDimension('y', ny)
zset = dataset.createDimension('z', nz)
time = dataset.createDimension('t', None)

#create dimensional variables
xs = dataset.createVariable('x', np.float64, ('x',))
ys = dataset.createVariable('y', np.float64, ('y',))
zs = dataset.createVariable('z', np.float64, ('z',))
ts = dataset.createVariable('t', np.float64, ('t',))

try:
    dataList = []
    for i in range(len(tasks)):
        dataList.append(np.array(hf['tasks/%s' % tasks[i]]))
except:
    sys.exit("There was an error putting your data into arrays")

#create variables that vary in x,y,z, and t
varList = []
for i in range(len(tasks)):
    varList.append(dataset.createVariable(tasks[i], np.float64, ('t','x','y','z')))

#assign dimensional data
xs[:] = x
ys[:] = y
zs[:] = z
ts[:] = t

#assign all other variables
try:
    for i in range(len(varList)):
        varList[i][:] = dataList[i]
except:
    sys.exit("There was an error writing the data arrays into netCDF variables")

dataset.close()


#declare variables
#eg. var1:var2:var2
taskString = ':'.join(tasks)
'''
#declare extents
if(sys.argv[3] != None):
    extents = sys.argv[3]
    print(sys.argv[3])
else:
    extents = '0:0:0:1:1:1'
'''
#I changed it
extents = '0:0:0:1:1:1'

#declare file names
ncd = sys.argv[2]+'.nc'
vdf = sys.argv[2]+'.vdf'


# The following lines need to be updated because Vapor is updated from 2.6.0 to 3.2.0
format = (taskString, taskString, nt, extents, ncd, vdf)
os.system('ncdfvdfcreate -vars %s -vars3d %s -numts %d -timedims t \
-timevars t -extents %s %s %s' %  format )
#example call: ncdfvdfcreate -vars Bx:By:Bz -vars3d Bx:By:Bz -numts 20 \
#   -timedimes t -timevars t -extents 0:-1:-1:10:1:1 ssx.nc ssx.vdf

format = (taskString, nt, ncd, vdf)
os.system('ncdf2vdf -vars %s -numts %d -timedims t %s %s' % format)
#example call: ncdf2vdf -vars Bx:By:Bz -numts 20 -timedims t ssx.nc ssx.vdf
