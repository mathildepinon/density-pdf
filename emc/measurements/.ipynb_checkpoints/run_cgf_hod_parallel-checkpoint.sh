#!/bin/bash
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --constraint=cpu
#SBATCH -t 1:00:00
#SBATCH --output='/global/u2/m/mpinon/_sbatch/emulator_parallel_r15.log'

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
module swap pyrecon/mpi pyrecon/main 
module load parallel

srun parallel --jobs 50 /global/homes/m/mpinon/density/emc/measurements/run_cgf_hod.sh {} ::: $(seq 18 117) 