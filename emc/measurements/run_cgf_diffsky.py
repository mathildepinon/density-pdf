import os
import time
import numpy as np
import pickle 
import argparse
from pathlib import Path
import fitsio
from astropy.table import Table
import h5py
from pyrecon.utils import MemoryMonitor
from acm.estimators.galaxy_clustering import BaseEnvironmentEstimator, DensityFieldCumulants
from acm import setup_logging
from cosmoprimo.fiducial import AbacusSummit
from cosmoprimo.cosmology import Cosmology


def read_lrg(filename=None, data=None, apply_rsd=True, los='z', wrap=True):
    #data = Table.read(filename)
    if data is None:
        data = h5py.File(filename, 'r')
    #pos = data['pos']
    pos = data['data']['pos']
    hubble = 100 * cosmo.efunc(redshift)
    scale_factor = 1 / (1 + redshift)
    if apply_rsd:
        #vel = data['vel']
        vel = data['data']['vel']
        pos_rsd = pos + vel / (hubble * scale_factor)
        los_dict = {'x': 0, 'y': 1, 'z': 2}
        pos[:, los_dict[los]] = pos_rsd[:, los_dict[los]]
    #is_lrg = data["diffsky_isLRG"].astype(bool)
    is_lrg = data['data']["diffsky_isLRG"].astype(bool)
    if wrap:
        mask = np.all((pos >= 0) & (pos <= boxsize), axis=1)
        return pos[is_lrg & mask]
    return pos[is_lrg]    

if __name__ == '__main__':
    setup_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument("--r", type=float, default=10)
    parser.add_argument("--basesim", type=str, default='unit', choices=['unit', 'abacus'])
    parser.add_argument("--phase", type=int, default=1)
    parser.add_argument("--galsampled", type=str, default='mass_conc', choices=['mass_conc', 'mass'])
    parser.add_argument("--logbins", type=bool, default=False)
    args = parser.parse_args()

    redshift = 0.5
    boxsize = 1000 if args.basesim=='unit' else 2000
    if args.basesim == 'unit':
        cosmo = Cosmology(Omega_m=0.3089, h=0.6774, n_s=0.9667, sigma8=0.8147, engine='class')  # UNIT cosmology
    else:
        cosmo = AbacusSummit(0)
    if args.logbins:
        lda = -np.logspace(3, -2, 101)
    else:
        lda = np.concatenate([[-50, -40, -30, -20], np.arange(-10, -5, 1), np.arange(-5, 5, 0.1), np.arange(5, 11, 1)])
    delta_bins = np.linspace(-1, 10, 1101)
    cellsize = 4
    r = args.r

    tstart = time.time()

    with MemoryMonitor() as mem:

        mem()

        density = DensityFieldCumulants(boxsize=boxsize, boxcenter=boxsize/2, cellsize=cellsize)

        density_dir = '/pscratch/sd/m/mpinon/density/emc/{}/z{:.1f}/r{:.0f}'.format(args.basesim, redshift, r)
        try:
            os.mkdir(density_dir)
            print(f"Directory '{density_dir}' created successfully.")
        except FileExistsError:
            print(f"Directory '{density_dir}' already exists.")
            
        # read simulation
        data_dir = '/global/cfs/cdirs/desicollab/users/gbeltzmo/C3EMC/{}'.format('UNIT' if args.basesim=='unit' else 'Abacus')
        if args.basesim=='unit':
            simname = 'galsampled_diffsky_mock_67120_fixedAmp_{:03d}_{}_v0.3'.format(args.phase, args.galsampled)
        else:
            simname = 'galsampled_diffsky_mock_abacus'
        data_fn = os.path.join(data_dir, simname+'.hdf5')
        data = h5py.File(data_fn, 'r') # read data only once because it can take a long time for abacus-based mock

        for los in ['x', 'y', 'z']:
            # load data
            t0 = time.time()
            data_positions = read_lrg(data=data, apply_rsd=True, los=los)
            t1 = time.time()
            print('Catalog loaded in elapsed time: {:.2f} s'.format(t1 - t0))

            print('Catalog size: {}'.format(data_positions.shape[0]))
            print('boxsize:', np.max(data_positions[:, 0])-np.min(data_positions[:, 0]))
            print('boxcenter:', (np.max(data_positions[:, 0])+np.min(data_positions[:, 0]))/2)

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
            pdf_fn = Path(density_dir) / 'density_pdf_cellsize{}{}_{}_los{}.npy'.format(cellsize, '_r{:.0f}'.format(r) if r else '', simname, los)
            cgf_fn = Path(density_dir) / 'density_cgf_cellsize{}{}_{}_los{}{}.npy'.format(cellsize, '_r{:.0f}'.format(r) if r else '', simname, los, '_logbins' if args.logbins else '')

            with open(pdf_fn, 'wb') as f:
                pickle.dump(pdf_dict, f)
            print('Saved PDF file: {}.'.format(pdf_fn))
            with open(cgf_fn, 'wb') as f:
                pickle.dump(cgf_dict, f)
            print('Saved CGF file: {}.'.format(cgf_fn))

    print("GCF and PDF computed for one mock for the 3 los in elapsed time {:.2f} s.".format(time.time()-tstart))