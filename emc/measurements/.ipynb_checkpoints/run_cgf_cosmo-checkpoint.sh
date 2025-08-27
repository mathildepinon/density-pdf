#!/bin/bash

#source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
#module swap pyrecon/mpi pyrecon/main 

python /global/u2/m/mpinon/density/emc/measurements/run_cgf_cosmo.py --start_cosmo $2 --n_cosmo 1 --start_hod $3 --n_hod 1 --r $1