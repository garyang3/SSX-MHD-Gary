"""2D Hartmann flow example
Uses vector potential form of MHD; incompressible Navier-Stokes, no further approximations.
"""
import os
import sys
import numpy as np
import pathlib
import h5py
import time
import dedalus.public as de
from dedalus.extras import flow_tools
from dedalus.tools import post
from mpi4py import MPI


import logging
logger = logging.getLogger(__name__)


Lx = 10.
Ly = 1
Lz = 0.5
nx = 20
ny = 20
nz = 40


Ha = 10. # change this parameter
Re = 1.
Rm = 1.
Pi = 1.
tau = 0.1

mesh = [2,2]
stop_time = 4
data_dir = "checkpoints/scratch"

# bases and domain
x = de.Fourier('x', nx, interval=[0,Lx], dealias=3/2)
y = de.Fourier('y', ny, interval=[-Ly, Ly], dealias=3/2)
z = de.Chebyshev('z', nz, interval=[-Lz,Lz], dealias=3/2) #the coupled dimension
domain = de.Domain([x,y,z],grid_dtype='float', mesh=mesh)

# variables and parameters
hartmann = de.IVP(domain, variables=['vx', 'vy', 'vz' ,'Ay', 'p', 'vx_z', 'vy_z', 'vz_z', 'Ay_z'])
hartmann.parameters['Ha'] = Ha
hartmann.parameters['Re'] = Re
hartmann.parameters['Rm'] = Rm
hartmann.parameters['Pi'] = Pi # pressure gradient driving flow
hartmann.parameters['Lx'] = Lx
hartmann.parameters['Lz'] = Lz
hartmann.parameters['tau'] = tau
hartmann.substitutions['Bx'] = "Ay_z"
hartmann.substitutions['Bz'] = "dx(Ay) + 1."
hartmann.substitutions['Lap(A, Az)'] = "dx(dx(A)) + dy(dy(A)) + dz(Az)"
hartmann.substitutions['Jy'] = "-Lap(Ay, Ay_z)"
hartmann.substitutions['Avg_x(A)'] = "integ(A,'x')/Lx"

# Navier Stokes
hartmann.add_equation("dt(vx) + dx(p) - Lap(vx, vx_z)/Re = -vx*dx(vx) -vy*dy(vx) - vz*dz(vz) - Ha**2/(Re*Rm) * Jy*Bz - Pi*(exp(-t/tau) - 1)")
hartmann.add_equation("dt(vy) + dy(p) - Lap(vy, vy_z)/Re = -vx*dx(vy) -vy*dy(vy) - vz*dz(vy)")
hartmann.add_equation("dt(vz) + dz(p) - Lap(vz, vz_z)/Re = -vz*dx(vz) -vy*dy(vz) - vz*dz(vz) + Ha**2/(Re*Rm) * Jy*Bx")

# div(v) = 0, incompressible
hartmann.add_equation("dx(vx) + dy(vy) + vz_z = 0")

# Az
hartmann.add_equation("dt(Ay) - Lap(Ay, Ay_z)/Rm = vx*Bz - vz*Bx")

# first order form
hartmann.add_equation("dz(Ay) - Ay_z = 0")
hartmann.add_equation("dz(vx) - vx_z = 0")
hartmann.add_equation("dz(vy) - vy_z = 0")
hartmann.add_equation("dz(vz) - vz_z = 0")

# boundary conditions: nonslip at wall, pressure concentrated on the left
hartmann.add_bc("left(vx) = 0")
hartmann.add_bc("right(vx) = 0")
hartmann.add_bc("left(vy) = 0")
hartmann.add_bc("right(vy) = 0")
hartmann.add_bc("left(vz) = 0")
hartmann.add_bc("right(vz) = 0", condition="(nx != 0)")
hartmann.add_bc("right(p) = 0", condition="(nx == 0)")
hartmann.add_bc("left(Ay) = 0")
hartmann.add_bc("right(Ay) = 0")

# build solver
solver = hartmann.build_solver(de.timesteppers.MCNAB2)
logger.info("Solver built")

# Integration parameters
solver.stop_sim_time = stop_time
solver.stop_wall_time = 5*24*60.*60
solver.stop_iteration = np.inf
dt = 1e-3

# Initial conditions are zero by default in all fields

# Analysis
analysis_tasks = []
check = solver.evaluator.add_file_handler(os.path.join(data_dir,'checkpoints'), wall_dt=3540, max_writes=50)
check.add_system(solver.state)
analysis_tasks.append(check)

snap = solver.evaluator.add_file_handler(os.path.join(data_dir,'snapshots'), sim_dt=1e-1, max_writes=200)
snap.add_task("Bx", scales=1)
snap.add_task("Bz", scales=1)
snap.add_task("Ay", scales=1)
snap.add_task("vx", scales=1)
snap.add_task("vy", scales=1)
snap.add_task("vz", scales=1)

analysis_tasks.append(snap)

integ = solver.evaluator.add_file_handler(os.path.join(data_dir,'integrals'), sim_dt=1e-3, max_writes=200)
integ.add_task("Avg_x(vx)", name='vx_x', scales=1)
integ.add_task("Avg_x(vy)", name='vy_x', scales=1)
integ.add_task("Avg_x(vz)", name='vz_x', scales=1)
integ.add_task("Avg_x(Bx)", name='Bx_x', scales=1)
integ.add_task("Avg_x(Bz)", name='Bz_x', scales=1)
analysis_tasks.append(integ)

timeseries = solver.evaluator.add_file_handler(os.path.join(data_dir,'timeseries'), sim_dt=1e-2)
timeseries.add_task("0.5*integ(vx**2 + vy**2 + vz**2)",name='Ekin')
timeseries.add_task("0.5*integ(Bx**2 + Bz**2)",name='Emag')
analysis_tasks.append(timeseries)

# Flow properties
flow = flow_tools.GlobalFlowProperty(solver, cadence=1)
flow.add_property("0.5*(vx**2 + vy**2 + vz**2)", name='Ekin')

try:
    logger.info('Starting loop')
    start_run_time = time.time()

    while solver.ok:
        solver.step(dt)
        if (solver.iteration-1) % 1 == 0:
            logger.info('Iteration: %i, Time: %e, dt: %e' %(solver.iteration, solver.sim_time, dt))
            logger.info('Max E_kin = %17.12e' %flow.max('Ekin'))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    end_run_time = time.time()
    logger.info('Iterations: %i' %solver.iteration)
    logger.info('Sim end time: %f' %solver.sim_time)
    logger.info('Run time: %f' %(end_run_time-start_run_time))


logger.info('beginning join operation')
for task in analysis_tasks:
    logger.info(task.base_path)
    post.merge_analysis(task.base_path)
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
if rank == 0:
    dir_path = os.path.dirname(os.path.realpath(__file__))
    print(dir_path)
    if os.path.isfile(dir_path+"/scratch/integrals/integrals.h5"):
        os.remove(dir_path+"/scratch/integrals/integrals.h5")
set_paths = list(pathlib.Path('scratch/integrals').glob("*.h5"))
post.merge_sets('checkpoints/scratch/integrals/integrals.h5', set_paths)