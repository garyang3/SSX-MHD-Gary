This directory contains modified files I inherited from Carlos Cartagena-Sanchez and Ercong Luo. 

**Gary_Final_Reg_Co**.job File is used to submit a batch job to the Bridges-2 cluster. It contains activation of the dedalus environment and a mpi parrallelization of dedalus.

**SSX_model_A_2_spheromaks.py** is the most important simulation file that advances the simulation with calculated time step using CFL condition; the initial condition is generated using two_spheromaks.py. 
We save files in hdf5 format using the "solver.evaluator.add_file_handler" method. To interpoloate or integrate (differentiate) at some point or some argument, simply use "interp(integ(T,'x','y'),z=0.4125)". 

The main update from last year is the correction in the momentum equation "dt(vx) + dx(T) - nu*Lap(vx) = -T*dx(lnrho) - vdotgrad(vx) + (jy*Bz - jz*By)/rho" and addition of file_handlers.

Line 194 to 240 contains initialization of density and temperature profile written by my advisor Michael Brown and Jeff Oishi (Bates College Professor). 

**SSX_model_A_load.py** also advances the simulation, but its initial condition can be a hdf5 file, generated from the "load_data" file handler in SSX_model_A_2_spheromaks.py. This is done by this:
'''
basedir = os.path.dirname(os.path.realpath(__file__))
dfile = basedir + '/load_data_s1.h5'
try:
   write, dt = solver.load_state(dfile, -1)
except IOError:
    sys.exit("Your input file could not be accessed")
'''

One flaw in this file is that it cannot clip lnrho and T in the main loop. I have no idea why this is so. 


**two_spheromaks.py** Initializes Ax,Ay,Az for a spheromaks solving equations can be found in any book about spheromak using Bessel's function. Spheromak_A is the main function in the file, and the auxillary function getS and getS1 gives two helicity of the spheromaks. 
