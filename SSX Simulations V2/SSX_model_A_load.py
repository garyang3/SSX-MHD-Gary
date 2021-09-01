"""SSX_model_A.py
Usage:
    SSX_model_A_load.py <fileIn> <initStep>
This is the *simplest* model we will consider for modelling spheromaks evolving in the SSX wind tunnel.
Major simplificiations fall in two categories
Geometry
--------
We consider a square duct using parity bases (sin/cos) in all directions.
Equations
---------
The equations themselves are those from Schaffner et al (2014), with the following simplifications
* hall term off
* constant eta instead of Spitzer
* no wall recycling term
* no mass diffusion
For this first model, rather than kinematic viscosity nu and thermal
diffusivitiy chi varying with density rho as they should, we are here
holding them *constant*. This dramatically simplifies the form of the
equations in Dedalus.
We use the vector potential, and enforce the Coulomb Gauge, div(A) = 0.
"""

import os
import sys
import time
import numpy as np
import h5py

import dedalus.public as de
from dedalus.extras import flow_tools

from matplotlib import pyplot
import matplotlib as mpl


import logging
logger = logging.getLogger(__name__)


# for optimal efficiency: nx should be divisible by mesh[0], ny by mesh[1], and
# nx should be close to ny. Bridges nodes have 28 cores, so mesh[0]*mesh[1]
# should be a multiple of 28.
nx = 64
ny = 64
nz = 360
r = 1
length = 10

# for 3D runs, you can divide the work up over two dimensions (x and y).
# The product of the two elements of mesh *must* equal the number
# of cores used.
mesh = [8,16]

kappa = 0.01
mu = 0.05
eta = 0.001
rho0 = 1
gamma = 5./3.
eta_sp = 2.7*10**(-4)
eta_ch = 4.4 * 10**(-3)
v0_ch = 2.9 * 10**(-2)
cor_fac = 144094#(0.5^2/(4*pi))/(1.38064852*10^(-23)*10^(16)) = (B^2/(4pi))/(k*n)
kel_to_ev = 0.00008617

x = de.SinCos('x', nx, interval=(-r, r))
y = de.SinCos('y', ny, interval=(-r, r))
z = de.SinCos('z', nz, interval=(0,length))

domain = de.Domain([x,y,z],grid_dtype='float', mesh=mesh)

SSX = de.IVP(domain, variables=['lnrho','T', 'vx', 'vy', 'vz', 'Ax', 'Ay', 'Az', 'phi'])

SSX.meta['T','lnrho']['x', 'y', 'z']['parity'] = 1
SSX.meta['phi']['x', 'y', 'z']['parity'] = -1

SSX.meta['vx']['y', 'z']['parity'] =  1
SSX.meta['vx']['x']['parity'] = -1
SSX.meta['vy']['x', 'z']['parity'] = 1
SSX.meta['vy']['y']['parity'] = -1
SSX.meta['vz']['x', 'y']['parity'] = 1
SSX.meta['vz']['z']['parity'] = -1

SSX.meta['Ax']['x']['parity'] = 1
SSX.meta['Ax']['y']['parity'] =  -1
SSX.meta['Ax']['z']['parity'] =  -1
SSX.meta['Ay']['x', 'z']['parity'] = -1
SSX.meta['Ay']['y']['parity'] = 1
SSX.meta['Az']['x', 'y']['parity'] = -1
SSX.meta['Az']['z']['parity'] = 1

SSX.parameters['mu'] = mu
SSX.parameters['chi'] = kappa/rho0
SSX.parameters['nu'] = mu/rho0
SSX.parameters['eta'] = eta
SSX.parameters['gamma'] = gamma
SSX.parameters['eta_sp'] = eta_sp
SSX.parameters['eta_ch'] = eta_ch
SSX.parameters['v0_ch'] = v0_ch
SSX.parameters['cor_fac'] = cor_fac
SSX.parameters['kel_to_ev'] = kel_to_ev

SSX.substitutions['divv'] = "dx(vx) + dy(vy) + dz(vz)"
SSX.substitutions['vdotgrad(A)'] = "vx*dx(A) + vy*dy(A) + vz*dz(A)"
SSX.substitutions['Bdotgrad(A)'] = "Bx*dx(A) + By*dy(A) + Bz*dz(A)"
SSX.substitutions['Lap(A)'] = "dx(dx(A)) + dy(dy(A)) + dz(dz(A))"
SSX.substitutions['Bx'] = "dy(Az) - dz(Ay)"
SSX.substitutions['By'] = "dz(Ax) - dx(Az)"
SSX.substitutions['Bz'] = "dx(Ay) - dy(Ax)"

# Coulomb Gauge implies J = -Laplacian(A)
SSX.substitutions['jx'] = "-Lap(Ax)"
SSX.substitutions['jy'] = "-Lap(Ay)"
SSX.substitutions['jz'] = "-Lap(Az)"
SSX.substitutions['J2'] = "jx**2 + jy**2 + jz**2"
SSX.substitutions['rho'] = "exp(lnrho)"
#SSX.substitutions['eta_s'] = "eta_sp/(sqrt(cor_fac*kel_to_ev*T)**3)"
# CFL substitutions
SSX.substitutions['Va_x'] = "Bx/sqrt(rho)"
SSX.substitutions['Va_y'] = "By/sqrt(rho)"
SSX.substitutions['Va_z'] = "Bz/sqrt(rho)"
SSX.substitutions['Cs'] = "sqrt(gamma*T)"

# Continuity
SSX.add_equation("dt(lnrho) + divv = - vdotgrad(lnrho)")

# Momentum
SSX.add_equation("dt(vx) + dx(T) - nu*Lap(vx) = -T*dx(lnrho) - vdotgrad(vx) + (jy*Bz - jz*By)/rho")
SSX.add_equation("dt(vy) + dy(T) - nu*Lap(vy) = -T*dy(lnrho) - vdotgrad(vy) + (jz*Bx - jx*Bz)/rho")
SSX.add_equation("dt(vz) + dz(T) - nu*Lap(vz) = -T*dz(lnrho) - vdotgrad(vz) + (jx*By - jy*Bx)/rho")

# MHD equations: A
SSX.add_equation("dt(Ax) + dx(phi) = - eta*jx + vy*Bz - vz*By")
SSX.add_equation("dt(Ay) + dy(phi) = - eta*jy + vz*Bx - vx*Bz")
SSX.add_equation("dt(Az) + dz(phi) = - eta*jz + vx*By - vy*Bx")
SSX.add_equation("dx(Ax) + dy(Ay) + dz(Az) = 0", condition = "(nx != 0) or (ny != 0) or (nz != 0)")
SSX.add_equation("phi = 0", condition = "(nx == 0) and (ny == 0) and (nz == 0)")


# Energy
SSX.add_equation("dt(T) - (gamma - 1) * chi*Lap(T) = - (gamma - 1) * T * divv  - vdotgrad(T) + (gamma - 1)*eta*J2")

solver = SSX.build_solver(de.timesteppers.RK443)

# Initial timestep
dt = 1e-6

# Integration parameters
solver.stop_sim_time = 50
solver.stop_wall_time = 60*60*20
solver.stop_iteration = np.inf

#loading past simulation
basedir = os.path.dirname(os.path.realpath(__file__))
dfile = basedir + '/load_data_s1.h5'
try:
   write, dt = solver.load_state(dfile, -1)
except IOError:
    sys.exit("Your input file could not be accessed")

wall_dt_checkpoints = 60*55
output_cadence = 0.1 # This is in simulation time units

'''checkpoint = solver.evaluator.add_file_handler('checkpoints2', max_writes=1, wall_dt=wall_dt_checkpoints, mode='overwrite')
    checkpoint.add_system(solver.state, layout='c')'''

field_writes = solver.evaluator.add_file_handler('fields', max_writes = 500, sim_dt = output_cadence, mode = 'append')
field_writes.add_task('vx')
field_writes.add_task('vy')
field_writes.add_task('vz')
field_writes.add_task('Bx')
field_writes.add_task('By')
field_writes.add_task('Bz')
field_writes.add_task('jx')
field_writes.add_task('jy')
field_writes.add_task('jz')
field_writes.add_task("exp(lnrho)", name = 'rho')
field_writes.add_task('T')
#field_writes.add_task('eta1')

parameter_writes = solver.evaluator.add_file_handler('parameters', max_writes = 1, sim_dt = output_cadence, mode = 'append')
parameter_writes.add_task('mu')
parameter_writes.add_task('eta')
parameter_writes.add_task('nu')
parameter_writes.add_task('chi')
parameter_writes.add_task('gamma')

load_writes = solver.evaluator.add_file_handler('load_data', max_writes = 500, sim_dt = output_cadence, mode = 'append')
load_writes.add_task('vx')
load_writes.add_task('vy')
load_writes.add_task('vz')
load_writes.add_task('Ax')
load_writes.add_task('Ay')
load_writes.add_task('Az')
load_writes.add_task('lnrho')
load_writes.add_task('T')
load_writes.add_task('phi')

slice2d_writes = solver.evaluator.add_file_handler('2dslices', max_writes=500, sim_dt=output_cadence, mode='append')
slice2d_writes.add_task("integ(exp(lnrho),'z')", name="integral_density_over_z")
slice2d_writes.add_task("integ(T,'z')", name="integral_temp_over_z")
slice2d_writes.add_task("integ(exp(lnrho),'x')", name="integral_density_over_x")
slice2d_writes.add_task("integ(T,'x')", name="integral_temp_over_x")
slice2d_writes.add_task("integ(exp(lnrho),'y')", name="integral_density_over_y")
slice2d_writes.add_task("integ(T,'y')", name="integral_temp_over_y")
slice2d_writes.add_task("interp(T,z=0)", name="T_at_z=0")
slice2d_writes.add_task("interp(exp(lnrho),z=0)", name="rho_at_z=0")
slice2d_writes.add_task("interp(T,x=0)", name="T_at_x=0")
slice2d_writes.add_task("interp(exp(lnrho),x=0)", name="rho_at_x=0")
slice2d_writes.add_task("interp(T,y=0)", name="T_at_y=0")
slice2d_writes.add_task("interp(exp(lnrho),y=0)", name="rho_at_y=0")

slice1d_writes = solver.evaluator.add_file_handler('1dslices', max_writes=500, sim_dt=output_cadence, mode='append')
#integrate over xy
slice1d_writes.add_task("integ(exp(lnrho),'x','y')", name="integral_density_over_xy")
slice1d_writes.add_task("integ(T,'x','y')", name="integral_temp_over_xy")
slice1d_writes.add_task("interp(integ(T,'x','y'),z=0)", name="integral_temp_over_xy_at_z=0")
slice1d_writes.add_task("interp(integ(exp(lnrho),'x','y'),z=0)", name="integral_density_overxy_at_z=0")
slice1d_writes.add_task("interp(integ(exp(lnrho),'x','y'),z=0.4125)", name="integral_density_overxy_at_z=probe_1_z")
slice1d_writes.add_task("interp(integ(T,'x','y'),z=0.4125)", name="integral_temp_over_xy_at_z=probe_1_z")
slice1d_writes.add_task("interp(integ(exp(lnrho),'x','y'),z=-0.4125)", name="integral_density_overxy_at_z=probe_2_z")
slice1d_writes.add_task("interp(integ(T,'x','y'),z=-0.4125)", name="integral_temp_over_xy_at_z=probe_2_z")
slice1d_writes.add_task("interp(integ(exp(lnrho),'x','y'),z=-0.825)", name="integral_density_overxy_at_z=probe_3_z")
slice1d_writes.add_task("interp(integ(T,'x','y'),z=-0.825)", name="integral_temp_over_xy_at_z=probe_3_z")
slice1d_writes.add_task("interp(integ(exp(lnrho),'x','y'),z=0.825)", name="integral_density_overxy_at_z=probe_4_z")
slice1d_writes.add_task("interp(integ(T,'x','y'),z=0.825)", name="integral_temp_over_xy_z=probe_4_z")
#integrate over yz
slice1d_writes.add_task("integ(exp(lnrho),'y','z')", name="integral_density_over_yz")
slice1d_writes.add_task("integ(T,'y','z')", name="integral_temp_over_yz")
slice1d_writes.add_task("interp(integ(T,'y','z'),x=0)", name="integral_temp_over_yz_at_x=probe_1_x")
slice1d_writes.add_task("interp(integ(exp(lnrho),'y','z'),x=0)", name="integral_density_overxy_at_x=probe_1_x")
slice1d_writes.add_task("interp(integ(T,'y','z'),x=0.475)", name="integral_temp_over_yz_at_x=probe_2_x")
slice1d_writes.add_task("interp(integ(exp(lnrho),'y','z'),x=0.475)", name="integral_density_overxy_at_x=probe_2_x")
slice1d_writes.add_task("interp(integ(T,'y','z'),x=0.95)", name="integral_temp_over_yz_at_x=probe_3_x")
slice1d_writes.add_task("interp(integ(exp(lnrho),'y','z'),x=0.95)", name="integral_density_overxy_at_x=probe_3_x")
slice1d_writes.add_task("interp(integ(T,'y','z'),x=1.425)", name="integral_temp_over_yz_at_x=probe_4_x")
slice1d_writes.add_task("interp(integ(exp(lnrho),'y','z'),x=1.425)", name="integral_density_overxy_at_x=probe_4_x")


helicity_writes = solver.evaluator.add_file_handler('helicity', max_writes=500, sim_dt=output_cadence, mode='append')
helicity_writes.add_task("integ(Ax*Bx+Ay*By+Az*Bz,'x','y','z')", name="total_helicity")
helicity_writes.add_task("Ax*Bx+Ay*By+Az*Bz", name="helicity_at_pos")

#field_at_pos_max_writes = file_handling.getfloat('field_at_pos_max_writes')
field_at_pos_writes = solver.evaluator.add_file_handler('bfield', max_writes=500, sim_dt=output_cadence, mode='append')
field_at_pos_writes.add_task("interp(Bx,x=0,z=0.4125,y=0)", name="bx_at_x=probe_1_x_z=probe_1_z")
field_at_pos_writes.add_task("interp(By,x=0,z=0.4125,y=0)", name="by_at_x=probe_1_x_z=probe_1_z")
field_at_pos_writes.add_task("interp(Bz,x=0,z=0.4125,y=0)", name="bz_at_x=probe_1_x_z=probe_1_z")
field_at_pos_writes.add_task("interp(Bx,x=0,z=-0.4125,y=0)", name="bx_at_x=probe_1_x_z=probe_2_z")
field_at_pos_writes.add_task("interp(By,x=0,z=-0.4125,y=0)", name="by_at_x=probe_1_x_z=probe_2_z")
field_at_pos_writes.add_task("interp(Bz,x=0,z=-0.4125,y=0)", name="bz_at_x=probe_1_x_z=probe_2_z")
field_at_pos_writes.add_task("interp(Bx,x=0,z=-0.825,y=0)", name="bx_at_x=probe_1_x_z=probe_3_z")
field_at_pos_writes.add_task("interp(By,x=0,z=-0.825,y=0)", name="by_at_x=probe_1_x_z=probe_3_z")
field_at_pos_writes.add_task("interp(Bz,x=0,z=-0.825,y=0)", name="bz_at_x=probe_1_x_z=probe_3_z")
field_at_pos_writes.add_task("interp(Bx,x=0,z=0.825,y=0)", name="bx_at_x=probe_1_x_z=probe_4_z")
field_at_pos_writes.add_task("interp(By,x=0,z=0.825,y=0)", name="by_at_x=probe_1_x_z=probe_4_z")
field_at_pos_writes.add_task("interp(Bz,x=0,z=0.825,y=0)", name="bz_at_x=probe_1_x_z=probe_4_z")
##
field_at_pos_writes.add_task("interp(Bx,x=0.475,z=0.4125,y=0)", name="bx_at_x=probe_2_x_z=probe_1_z")
field_at_pos_writes.add_task("interp(By,x=0.475,z=0.4125,y=0)", name="by_at_x=probe_2_x_z=probe_1_z")
field_at_pos_writes.add_task("interp(Bz,x=0.475,z=0.4125,y=0)", name="bz_at_x=probe_2_x_z=probe_1_z")
field_at_pos_writes.add_task("interp(Bx,x=0.475,z=-0.4125,y=0)", name="bx_at_x=probe_2_x_z=probe_2_z")
field_at_pos_writes.add_task("interp(By,x=0.475,z=-0.4125,y=0)", name="by_at_x=probe_2_x_z=probe_2_z")
field_at_pos_writes.add_task("interp(Bz,x=0.475,z=-0.4125,y=0)", name="bz_at_x=probe_2_x_z=probe_2_z")
field_at_pos_writes.add_task("interp(Bx,x=0.475,z=-0.825,y=0)", name="bx_at_x=probe_2_x_z=probe_3_z")
field_at_pos_writes.add_task("interp(By,x=0.475,z=-0.825,y=0)", name="by_at_x=probe_2_x_z=probe_3_z")
field_at_pos_writes.add_task("interp(Bz,x=0.475,z=-0.825,y=0)", name="bz_at_x=probe_2_x_z=probe_3_z")
field_at_pos_writes.add_task("interp(Bx,x=0.475,z=0.825,y=0)", name="bx_at_x=probe_2_x_z=probe_4_z")
field_at_pos_writes.add_task("interp(By,x=0.475,z=0.825,y=0)", name="by_at_x=probe_2_x_z=probe_4_z")
field_at_pos_writes.add_task("interp(Bz,x=0.475,z=0.825,y=0)", name="bz_at_x=probe_2_x_z=probe_4_z")
##
field_at_pos_writes.add_task("interp(Bx,x=0.95,z=0.4125,y=0)", name="bx_at_x=probe_3_x_z=probe_1_z")
field_at_pos_writes.add_task("interp(By,x=0.95,z=0.4125,y=0)", name="by_at_x=probe_3_x_z=probe_1_z")
field_at_pos_writes.add_task("interp(Bz,x=0.95,z=0.4125,y=0)", name="bz_at_x=probe_3_x_z=probe_1_z")
field_at_pos_writes.add_task("interp(Bx,x=0.95,z=-0.4125,y=0)", name="bx_at_x=probe_3_x_z=probe_2_z")
field_at_pos_writes.add_task("interp(By,x=0.95,z=-0.4125,y=0)", name="by_at_x=probe_3_x_z=probe_2_z")
field_at_pos_writes.add_task("interp(Bz,x=0.95,z=-0.4125,y=0)", name="bz_at_x=probe_3_x_z=probe_2_z")
field_at_pos_writes.add_task("interp(Bx,x=0.95,z=-0.825,y=0)", name="bx_at_x=probe_3_x_z=probe_3_z")
field_at_pos_writes.add_task("interp(By,x=0.95,z=-0.825,y=0)", name="by_at_x=probe_3_x_z=probe_3_z")
field_at_pos_writes.add_task("interp(Bz,x=0.95,z=-0.825,y=0)", name="bz_at_x=probe_3_x_z=probe_3_z")
field_at_pos_writes.add_task("interp(Bx,x=0.95,z=0.825,y=0)", name="bx_at_x=probe_3_x_z=probe_4_z")
field_at_pos_writes.add_task("interp(By,x=0.95,z=0.825,y=0)", name="by_at_x=probe_3_x_z=probe_4_z")
field_at_pos_writes.add_task("interp(Bz,x=0.95,z=0.825,y=0)", name="bz_at_x=probe_3_x_z=probe_4_z")
##
field_at_pos_writes.add_task("interp(Bx,x=1.425,z=0.4125,y=0)", name="bx_at_x=probe_4_x_z=probe_1_z")
field_at_pos_writes.add_task("interp(By,x=1.425,z=0.4125,y=0)", name="by_at_x=probe_4_x_z=probe_1_z")
field_at_pos_writes.add_task("interp(Bz,x=1.425,z=0.4125,y=0)", name="bz_at_x=probe_4_x_z=probe_1_z")
field_at_pos_writes.add_task("interp(Bx,x=1.425,z=-0.4125,y=0)", name="bx_at_x=probe_4_x_z=probe_2_z")
field_at_pos_writes.add_task("interp(By,x=1.425,z=-0.4125,y=0)", name="by_at_x=probe_4_x_z=probe_2_z")
field_at_pos_writes.add_task("interp(Bz,x=1.425,z=-0.4125,y=0)", name="bz_at_x=probe_4_x_z=probe_2_z")
field_at_pos_writes.add_task("interp(Bx,x=1.425,z=-0.825,y=0)", name="bx_at_x=probe_4_x_z=probe_3_z")
field_at_pos_writes.add_task("interp(By,x=1.425,z=-0.825,y=0)", name="by_at_x=probe_4_x_z=probe_3_z")
field_at_pos_writes.add_task("interp(Bz,x=1.425,z=-0.825,y=0)", name="bz_at_x=probe_4_x_z=probe_3_z")
field_at_pos_writes.add_task("interp(Bx,x=1.425,z=0.825,y=0)", name="bx_at_x=probe_4_x_z=probe_4_z")
field_at_pos_writes.add_task("interp(By,x=1.425,z=0.825,y=0)", name="by_at_x=probe_4_x_z=probe_4_z")
field_at_pos_writes.add_task("interp(Bz,x=1.425,z=0.825,y=0)", name="bz_at_x=probe_4_x_z=probe_4_z")


# Flow properties
flow = flow_tools.GlobalFlowProperty(solver, cadence = 1)
flow.add_property("sqrt(vx*vx + vy*vy + vz*vz) / nu", name = 'Re_k')
flow.add_property("sqrt(vx*vx + vy*vy + vz*vz) / eta", name = 'Re_m')
flow.add_property("sqrt(vx*vx + vy*vy + vz*vz)", name = 'flow_speed')
flow.add_property("sqrt(vx*vx + vy*vy + vz*vz) / sqrt(T)", name = 'Ma')
flow.add_property("sqrt(Bx*Bx + By*By + Bz*Bz) / sqrt(rho)", name = 'Al_v')
flow.add_property("T", name = 'temp')

char_time = 1. # this should be set to a characteristic time in the problem (the alfven crossing time of the tube, for example)
CFL_safety = 0.3
CFL = flow_tools.CFL(solver, initial_dt = dt, cadence = 10, safety = CFL_safety,
                     max_change = 1.5, min_change = 0.005, max_dt = output_cadence, threshold = 0.05)
CFL.add_velocities(('vx', 'vy', 'vz'))
CFL.add_velocities(('Va_x', 'Va_y', 'Va_z'))
CFL.add_velocities(( 'Cs', 'Cs', 'Cs'))


good_solution = True
# Main loop
try:
    logger.info('Starting loop')
    logger_string = 'kappa: {:.3g}, mu: {:.3g}, eta: {:.3g}, dt: {:.3g}'.format(kappa, mu, eta, dt)
    logger.info(logger_string)
    start_time = time.time()
    while solver.ok and good_solution:
        dt = CFL.compute_dt()
        solver.step(dt)
        
        if (solver.iteration-1) % 1 == 0:
            logger_string = 'iter: {:d}, t/tb: {:.2e}, dt/tb: {:.2e}, sim_time: {:.4e}, dt: {:.2e}'.format(solver.iteration, solver.sim_time/char_time, dt/char_time, solver.sim_time, dt)
            #logger_string += 'min_rho: {:.4e}'.format(lnrho['g'].min())
            Re_k_avg = flow.grid_average('Re_k')
            Re_m_avg = flow.grid_average('Re_m')
            v_avg = flow.grid_average('flow_speed')
            Al_v_avg = flow.grid_average('Al_v')
            logger_string += ' Max Re_k = {:.2g}, Avg Re_k = {:.2g}, Max Re_m = {:.2g}, Avg Re_m = {:.2g}, Max vel = {:.2g}, Avg vel = {:.2g}, Max alf vel = {:.2g}, Avg alf vel = {:.2g}, Max Ma = {:.1g}'.format(flow.max('Re_k'), Re_k_avg, flow.max('Re_m'),Re_m_avg, flow.max('flow_speed'), v_avg, flow.max('Al_v'), Al_v_avg, flow.max('Ma'))
            logger.info(logger_string)
            #np.clip(lnrho['g'], -4.9, 2, out=lnrho['g'])
            #np.clip(T['g'], 0.0038, 1000, out=T['g'])
            if not np.isfinite(Re_k_avg):
                good_solution = False
                logger.info("Terminating run.  Trapped on Reynolds = {}".format(Re_k_avg))
            if not np.isfinite(Re_m_avg):
                good_solution = False
                logger.info("Terminating run. Trapped on magnetic Reynolds = {}".format(Re_m_avg))

except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    end_time = time.time()
    logger.info('Iterations: %i' %solver.iteration)
    logger.info('Sim end time: %f' %solver.sim_time)
    logger.info('Run time: %.2f sec' %(end_time-start_time))
    logger.info('Run time: %f cpu-hr' %((end_time-start_time)/60/60*domain.dist.comm_cart.size))
    logger.info('Iter/sec: {:g}'.format(solver.iteration/(end_time-start_time)))
