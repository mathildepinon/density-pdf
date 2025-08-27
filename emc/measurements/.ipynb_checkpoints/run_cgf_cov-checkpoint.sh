#!/bin/bash

#source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
#module swap pyrecon/mpi pyrecon/main 

#python /global/u2/m/mpinon/density/emc/measurements/run_cgf_cov.py --start_phase $1 --n_phase 1 --r 10
python /global/u2/m/mpinon/density/emc/measurements/run_cgf_cov.py --start_phase $1 --n_phase 1 --r 15
python /global/u2/m/mpinon/density/emc/measurements/run_cgf_cov.py --start_phase $1 --n_phase 1 --r 20