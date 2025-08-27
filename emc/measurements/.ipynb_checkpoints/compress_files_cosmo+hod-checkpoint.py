import os
import numpy as np
from pathlib import Path
import pandas
import fitsio
import argparse


def read_lhc(statistic='cgf', start_cosmo=0, n_cosmo=1, start_hod=0, n_hod=100, phase=0, r=10, return_sep=False, los='combine', filter_low_n=False, data_dir='/pscratch/sd/m/mpinon/density/cosmo/', logbins=False):
    lhc_y = []
    sep = []
    cosmo_idx = []
    cosmo_idx_fail = []
    lhc_x_all = []
    
    lhc_x_c000 = pandas.read_csv('/pscratch/sd/e/epaillas/emc/cosmo+hod_params/AbacusSummit_c000.csv')
    lhc_x_names = list(lhc_x_c000.columns)
    lhc_x_names = [name.replace(' ', '').replace('#', '') for name in lhc_x_names]

    for cosmo in range(start_cosmo, start_cosmo+n_cosmo):
        hod_idx = []
        hod_idx_fail = []
        
        try:
            for hod in range(start_hod, start_hod+n_hod):
                try:
                    # Combine line-of-sights
                    if los=='combine':
                        data_list = list()
                        for i, l in enumerate(['x', 'y', 'z']):
                            data_fn = Path(data_dir) / 'c{:03}_ph{:03d}/density_{}_cellsize4_r{:.0f}_hod{:03}_los{}{}.npy'.format(cosmo, phase, statistic, r, hod, l, '_logbins' if logbins else '')
                            data = np.load(data_fn, allow_pickle=True)
                            data_list.append(data)
                        edges = np.mean([data_list[i]['edges' if statistic=='pdf' else 'lambda'] for i in range(3)], axis=0)
                        stat = np.mean([data_list[i][statistic] for i in range(3)], axis=0)

                    # Just one line-of-sight
                    else:
                        data_fn = Path(data_dir) / 'c{:03}_ph{:03d}/density_{}_cellsize4_r{:.0f}_hod{:03}_los{}.npy'.format(cosmo, phase, statistic, r, hod, los, '_logbins' if logbins else '')
                        data = np.load(data_fn, allow_pickle=True)
                        edges, stat = data['edges' if statistic=='pdf' else 'lambda'], data[statistic]

                    if statistic=='pdf':
                        edges = (edges[1:]+edges[:-1])/2
                    else:
                        edges = edges

                    # Filter out low nbar
                    if filter_low_n:
                        mock_dir = '/pscratch/sd/e/epaillas/emc/hods/cosmo+hod/z0.5/yuan23_prior/c{:03d}_ph000/seed0'.format(cosmo)
                        mock_fn = Path(mock_dir) / 'hod{:03d}.fits'.format(hod)
                        hod = fitsio.read(mock_fn)
                        nbar = hod.size/2000**3
                        if (nbar < 0.0003):
                            pass
                        else:
                            sep.append(edges)
                            lhc_y.append(stat)
                    else:
                        sep.append(edges)
                        lhc_y.append(stat)
                    hod_idx.append(hod)

                except FileNotFoundError:
                    hod_idx_fail.append(hod)
                    print(data_fn, ' not found.')
                    print('Measurement for the {} of HOD {}, phase {} does not exist. Ignoring this measurement.'.format(statistic, hod, phase))

            lhc_x = pandas.read_csv('/pscratch/sd/e/epaillas/emc/cosmo+hod_params/AbacusSummit_c{:03d}.csv'.format(cosmo))
            lhc_x = lhc_x.values[np.array(hod_idx),:]
            lhc_x_all.append(lhc_x)
            
            if len(hod_idx_fail) > 0:
                print('Cosmology {:03d}: measurements for the following {} HOD indices were not found and ignored:'.format(cosmo, len(hod_idx_fail)))
                print(hod_idx_fail)
            cosmo_idx.append(cosmo)

        except FileNotFoundError:
            cosmo_idx_fail.append(cosmo)
            
    print('Measurements for {} cosmologies found. The following cosmologies were not found:'.format(len(cosmo_idx)))
    print(cosmo_idx_fail)
 
    sep = np.mean(sep, axis=0)
    lhc_y = np.array(lhc_y)
    lhc_x = np.concatenate(lhc_x_all)
    if return_sep:
        return sep, lhc_x, lhc_y, lhc_x_names
    return lhc_x, lhc_y

def rebin_pdf(x, y, n=2):
    x = np.mean(x.reshape((-1, n)), axis=1)
    y = np.mean(y.reshape((y.shape[0], -1, n)), axis=-1)
    return x, y


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--stat", type=str, default='pdf', choices=['pdf', 'cgf'])
    parser.add_argument("--r", type=float, default=10)
    parser.add_argument("--start_hod", type=int, default=0)
    parser.add_argument("--n_hod", type=int, default=1)
    parser.add_argument("--start_cosmo", type=int, default=0)
    parser.add_argument("--n_cosmo", type=int, default=1)
    parser.add_argument("--phase", type=int, default=0)
    parser.add_argument("--filter_low_n", type=bool, default=False)
    parser.add_argument("--logbins", type=bool, default=False)
    parser.add_argument("--cut", type=bool, default=False)
    args = parser.parse_args()
 
    # HOD realizations
    data_dir = '/pscratch/sd/m/mpinon/density/cosmo/r{:.0f}'.format(args.r)
    z, lhc_x, lhc_y, lhc_x_names = read_lhc(statistic=args.stat, start_cosmo=args.start_cosmo, n_cosmo=args.n_cosmo, start_hod=args.start_hod, n_hod=args.n_hod, phase=args.phase, r=args.r, return_sep=True, data_dir=data_dir, filter_low_n=args.filter_low_n, logbins=args.logbins)
    print(f'Loaded LHC with shape: {lhc_x.shape}, {lhc_y.shape}')
    save_dir = '/pscratch/sd/m/mpinon/density/training_sets/cosmo+hod/ph{:03d}/seed0'.format(args.phase)

    if args.cut:
        if args.stat == 'cgf':
            rebin = {'10': 4, '15': 8, '20': 8}
            z = z[::rebin[str(int(args.r))]]
            lhc_y = lhc_y[:, ::rebin[str(int(args.r))]]
            mask = {'10': (z >= -200) & (z <= -0.5), '15': (z >= -500) & (z <= -0.5), '20': (z >= -500) & (z <= -0.5)}
            z = z[mask[str(int(args.r))]]
            lhc_y = lhc_y[:, mask[str(int(args.r))]]
        elif args.stat == 'pdf':
            z, lhc_y = rebin_pdf(z, lhc_y, n=4)
            mask = {'10': (z >= -1) & (z <= 4), '15': (z > -0.95) & (z <= 2), '20': (z >= -0.76) & (z <= 1.4)}
            z = z[mask[str(int(args.r))]]
            lhc_y = lhc_y[:, mask[str(int(args.r))]]

    save_name = '{}_r{:.0f}_lhc{}{}{}.npy'.format(args.stat, args.r, '_filtered' if args.filter_low_n else '', '_logbins' if args.logbins else '', '_cut' if args.cut else '')
    save_fn = Path(save_dir) / save_name
    if args.stat == 'pdf':
        cout = {'delta': z, 'lhc_x': lhc_x, 'lhc_y': lhc_y, 'lhc_x_names': lhc_x_names}
    elif args.stat == 'cgf':
        cout = {'lambda': z, 'lhc_x': lhc_x, 'lhc_y': lhc_y, 'lhc_x_names': lhc_x_names}
    print('Saving compressed measurements at {}.'.format(save_fn))
    np.save(save_fn, cout)
