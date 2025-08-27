#!/bin/bash
#SBATCH --nodes=1
#SBATCH --account=desi_g
#SBATCH --ntasks-per-node=1
#SBATCH --constraint=gpu
#SBATCH --gpus 4
#SBATCH -q regular
#SBATCH -t 2:00:00
#SBATCH --output='/global/homes/m/mpinon/_sbatch/optimize_sunbird_r15.log'


cosmodesienv main

srun python /global/homes/m/mpinon/density/emulator/optimize_sunbird.py --stat pdf --r 15