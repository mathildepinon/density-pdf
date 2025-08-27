#!/bin/bash
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --constraint=cpu
#SBATCH -t 13:00:00
#SBATCH --output='/global/u2/m/mpinon/_sbatch/emulator_parallel_r20.log'

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
module swap pyrecon/mpi pyrecon/main 
module load parallel

srun parallel --jobs 50 /global/homes/m/mpinon/density/emc/measurements/run_cgf_cosmo.sh {1} {2} {3} ::: $1 ::: $(seq 0 13) ::: $(seq 100 349)
srun parallel --jobs 50 /global/homes/m/mpinon/density/emc/measurements/run_cgf_cosmo.sh {1} {2} {3} ::: $1 ::: $(seq 100 182) ::: $(seq 100 349)