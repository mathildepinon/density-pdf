#!/bin/bash

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
module swap pyrecon/mpi pyrecon/main 
module load parallel

#srun parallel --jobs 12 python /global/u2/m/mpinon/density/emc/measurements/run_cgf_diffsky.py --r {1} --phase {2} --galsampled {3} --logbins True ::: 10 15 20 ::: $(seq 1 2) ::: 'mass' 'mass_conc'

srun parallel --jobs 12 python /global/u2/m/mpinon/density/emc/measurements/process_diffsky.py --stat pdf --r {1} --phase {2} --galsampled {3} ::: 10 15 20 ::: $(seq 1 2) ::: 'mass' 'mass_conc'

#python /global/u2/m/mpinon/density/emc/measurements/run_cgf_diffsky.py --r 10 --basesim 'abacus'
#python /global/u2/m/mpinon/density/emc/measurements/run_cgf_diffsky.py --r 15 --basesim 'abacus'
#python /global/u2/m/mpinon/density/emc/measurements/run_cgf_diffsky.py --r 20 --basesim 'abacus'