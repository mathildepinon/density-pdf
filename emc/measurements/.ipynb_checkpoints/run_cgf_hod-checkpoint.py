import os
import time
import numpy as np
import pickle 
import argparse
from pathlib import Path
import fitsio
from pyrecon.utils import MemoryMonitor
from acm.estimators.galaxy_clustering import BaseEnvironmentEstimator, DensityFieldCumulants
from acm import setup_logging
from cosmoprimo.fiducial import AbacusSummit


def get_hod_positions(input_fn, los='z'):
    hod = fitsio.read(input_fn)
    pos = np.c_[hod['X'], hod['Y'], hod['Z']]
    hubble = 100 * cosmo.efunc(redshift)
    scale_factor = 1 / (1 + redshift)
    if los == 'x':
        pos[:, 0] += hod['VX'] / (hubble * scale_factor)
    elif los == 'y':
        pos[:, 1] += hod['VY'] / (hubble * scale_factor)
    elif los == 'z':
        pos[:, 2] += hod['VZ'] / (hubble * scale_factor)
    return pos

if __name__ == '__main__':
    setup_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument("--start_hod", type=int, default=0)
    parser.add_argument("--n_hod", type=int, default=1)
    parser.add_argument("--r", type=float, default=10)

    args = parser.parse_args()
    start_hod = args.start_hod
    n_hod = args.n_hod

    cosmo = AbacusSummit(0)
    redshift = 0.5

    # read some random galaxy catalog
    data_dir = '/pscratch/sd/e/epaillas/emc/hods/z0.5/yuan23_prior/c000_ph000/seed0/'

    boxsize = 2000.0
    #lda = np.linspace(-50, 10, 601)
    lda = np.concatenate([[-50, -40, -30, -20], np.arange(-10, -5, 1), np.arange(-5, 5, 0.1), np.arange(5, 11, 1)])
    delta_bins = np.linspace(-1, 10, 1101)
    cellsize = 4
    r = args.r

    density_dir = '/pscratch/sd/m/mpinon/density/hods/r{:.0f}'.format(r)
    try:
        os.mkdir(density_dir)
        print(f"Directory '{density_dir}' created successfully.")
    except FileExistsError:
        print(f"Directory '{density_dir}' already exists.")

    tstart = time.time()

    with MemoryMonitor() as mem:

        mem()

        density = DensityFieldCumulants(boxsize=boxsize, boxcenter=boxsize/2, cellsize=cellsize)

        for i in range(start_hod, start_hod+n_hod):
            data_fn = Path(data_dir) / 'hod{:03d}.fits'.format(i)

            for los in ['x', 'y', 'z']:
                # load data
                t0 = time.time()
                data_positions = get_hod_positions(data_fn, los=los)
                t1 = time.time()
                print('Catalog loaded in elapsed time: {:.2f} s'.format(t1 - t0))

                print('Catalog size: {}'.format(data_positions.shape[0]))

                mem()

                # compute density contrast
                density.assign_data(positions=data_positions, wrap=True, clear_previous=True)
                density.set_density_contrast(smoothing_radius=r, save_wisdom=False)
                t2 = time.time()
                print('Density contrast computed in elapsed time: {:.2f} s'.format(t2 - t1))
                
                mem()

                query_positions = density.get_query_positions(density.delta_mesh, method='randoms', nquery=density._size_data, seed=0)
                t3 = time.time()
                print('Got query positions in elapsed time: {:.2f} s'.format(t3 - t2))

                density_cgf = density.compute_cumulants(lda, query_positions=query_positions)
                delta =  density.delta_query
                print('CGF computed in elapsed time: {:.2f} s'.format(time.time() - t3))

                mem()

                # density pdf
                sigma = np.std(delta)
                density_pdf, edges = np.histogram(delta, bins=delta_bins, density=True)
                mem()
                
                pdf_dict = {'edges': edges, 'pdf': density_pdf, 'sigma': sigma}
                cgf_dict = {'lambda': lda, 'cgf': density_cgf}
                mem()

                # save result

                pdf_fn = Path(density_dir) / 'density_pdf_cellsize{}{}_hod{:03d}_los{}.npy'.format(cellsize, '_r{:.0f}'.format(r) if r else '', i, los)
                cgf_fn = Path(density_dir) / 'density_cgf_cellsize{}{}_hod{:03d}_los{}.npy'.format(cellsize, '_r{:.0f}'.format(r) if r else '', i, los)
                
                with open(pdf_fn, 'wb') as f:
                    pickle.dump(pdf_dict, f)
                print('Saved PDF file: {}.'.format(pdf_fn))
                with open(cgf_fn, 'wb') as f:
                    pickle.dump(cgf_dict, f)
                print('Saved CGF file: {}.'.format(cgf_fn))
    
    print("GCF and PDF computed for one mock for the 3 los in elapsed time {:.2f} s.".format(time.time()-tstart))