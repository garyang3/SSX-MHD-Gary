3D_Hmann.py is the file need to do the Hartmann problem. 
In order for this to run in may have to create a folder called scratch (or checkpoints) on your own.

kdv_burgers.py, rayleigh_benard.py, and kevin-helmholtz.py are from the official dedalus project repor examples
and the official dedalus project websites in order to check whether my runs produce the correct results. 

the plotting tool on the dedalus repo did not work so I have myownplotRB.py to plot data from rayleigh_benard problem. 

toVapor.py only works on vapor 2.6.0, so be sure to download that version. 

I did not include any .h5 files here, but that's what dedalus output is gonna be. 
