This directory contains necessary simulation files for spitzer resistivity. As of Sep. 1 2021 the simulation suffers from divergence of v^2 (magnetic and kinetic reynolds number). 

The spitzer resistivity has eta_sp*T^(-3/2), with a leading coefficient. 


**SSX_model_A_2_spheromaks_spitzer_v2_CFL.py** has a non-conservative diffusivity on eta as a part of the CFL condition.

**load_everything_CFL.py** has 


![image](https://user-images.githubusercontent.com/66159074/131730789-f5bae832-c6ce-4f2a-acc1-3dda0ac06b16.png)


as CFL conditions. Essentially, these are the terms that were operated by the laplacian. 
