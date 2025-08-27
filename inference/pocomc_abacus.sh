#!/bin/bash
#SBATCH --account=desi_g
#SBATCH -q shared
#SBATCH -t 02:00:00
#SBATCH --constraint=gpu
#SBATCH --gpus-per-task 1
#SBATCH -n 1
#SBATCH --output='/global/homes/m/mpinon/_sbatch/inference_pocomc_pk.log'

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main

python /global/homes/m/mpinon/density/inference/pocomc_abacus.py