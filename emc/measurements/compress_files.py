import numpy as np
from pathlib import Path
import pandas
import fitsio
import argparse


def read_lhc(statistic='cgf', start_hod=0, n_hod=100, start_phase=None, n_phase=1, r=10, return_sep=False, los='combine', filter_low_n=False, data_dir='/pscratch/sd/m/mpinon/density/hods', cosmo='c000', logbins='False'):
    lhc_y = []
    sep = []
    if start_phase is None:
        phase = 0
        start_phase = 0
        n_phase = 1
    else:
        phase = 1
    hod_idx = []
    hod_idx_fail = []
    phase_idx_fail = []
    for hod in range(start_hod, start_hod+n_hod):
        for ph in range(start_phase, start_phase+n_phase):
            try:
                if los=='combine':
                    data_list = list()
                    for i, l in enumerate(['x', 'y', 'z']):
                        data_fn = Path(data_dir) / 'density_{}_cellsize4_r{:.0f}{}_hod{:03}_los{}{}.npy'.format(statistic, r, '_ph{:04d}'.format(ph) if phase else '', hod, l, '_logbins' if logbins else '')
                        #print('Loading {}'.format(data_fn))
                        data = np.load(data_fn, allow_pickle=True)
                        data_list.append(data)
                    edges = np.mean([data_list[i]['edges' if statistic=='pdf' else 'lambda'] for i in range(3)], axis=0)
                    stat = np.mean([data_list[i][statistic] for i in range(3)], axis=0)
                else:
                    data_fn = Path(data_dir) / 'density_{}_cellsize4_r{:.0f}{}_hod{:03}_los{}{}.npy'.format(statistic, r, '_ph{:04d}'.format(ph) if phase else '', hod, los, '_logbins' if logbins else '')
                    data = np.load(data_fn, allow_pickle=True)
                    edges, stat = data['edges' if statistic=='pdf' else 'lambda'], data[statistic]
                if statistic=='pdf':
                    edges = (edges[1:]+edges[:-1])/2
                else:
                    edges = edges
                if (not phase) and filter_low_n:
                    mock_dir = '/pscratch/sd/e/epaillas/emc/hods/z0.5/yuan23_prior/c000_ph000/seed0/'
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
                print('Measurement for the {} of HOD {}, phase {} does not exist. Ignoring this measurement.'.format(statistic, hod, ph))

    print('Measurements for the following {} HOD indices were not found and ignored.'.format(len(np.unique(hod_idx_fail))))
    print(np.unique(hod_idx_fail))
 
    sep = np.mean(sep, axis=0)
    lhc_y = np.array(lhc_y)
    lhc_x = pandas.read_csv('/pscratch/sd/e/epaillas/emc/hod_params/yuan23/hod_params_yuan23_{}.csv'.format(cosmo))
    lhc_x_names = list(lhc_x.columns)
    lhc_x_names = [name.replace(' ', '').replace('#', '') for name in lhc_x_names]
    lhc_x = lhc_x.values[np.array(hod_idx),:]
    if return_sep:
        return sep, lhc_x, lhc_y, lhc_x_names
    return lhc_x, lhc_y

def rebin_pdf(x, y, n=2):
    x = np.mean(x.reshape((-1, n)), axis=1)
    y = np.mean(y.reshape((y.shape[0], -1, n)), axis=-1)
    return x, y


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--mocks", type=str, default='hod', choices=['hod', 'cosmo', 'cov'])
    parser.add_argument("--stat", type=str, default='pdf', choices=['pdf', 'cgf'])
    parser.add_argument("--r", type=float, default=10)
    parser.add_argument("--start_hod", type=int, default=0)
    parser.add_argument("--n_hod", type=int, default=1)
    parser.add_argument("--filter_low_n", type=bool, default=False)
    parser.add_argument("--logbins", type=bool, default=False)
    parser.add_argument("--cut", type=bool, default=False)
    args = parser.parse_args()
 
    if args.mocks == 'hod':
        # HOD realizations
        data_dir = '/pscratch/sd/m/mpinon/density/hods/r{:.0f}'.format(args.r)
        z, lhc_x, lhc_y, lhc_x_names = read_lhc(statistic=args.stat, start_hod=args.start_hod, n_hod=args.n_hod, r=args.r, return_sep=True, data_dir=data_dir, filter_low_n=args.filter_low_n)
        print(f'Loaded LHC with shape: {lhc_x.shape}, {lhc_y.shape}')
        save_dir = '/pscratch/sd/m/mpinon/density/training_sets/hod/c000_ph000/seed0'
        save_name = '{}_r{:.0f}_lhc{}.npy'.format(args.stat, args.r, '_filtered' if args.filter_low_n else '')
        save_fn = Path(save_dir) / save_name
        if args.stat == 'pdf':
            cout = {'delta': z, 'lhc_x': lhc_x, 'lhc_y': lhc_y, 'lhc_x_names': lhc_x_names}
        elif args.stat == 'cgf':
            cout = {'lambda': z, 'lhc_x': lhc_x, 'lhc_y': lhc_y, 'lhc_x_names': lhc_x_names}
        print('Saving compressed measurements at {}.'.format(save_fn))
        np.save(save_fn, cout)

    elif args.mocks == 'cov':
        # small box realizations for covariance estimation
        data_dir='/pscratch/sd/m/mpinon/density/small/r{:.0f}/hod466'.format(args.r)
        z, lhc_x, lhc_y, lhc_x_names = read_lhc(statistic=args.stat, start_hod=466, n_hod=1, start_phase=3000, n_phase=2000, r=args.r, return_sep=True, data_dir=data_dir, logbins=args.logbins)
        print(f'Loaded LHC with shape: {lhc_x.shape}, {lhc_y.shape}')

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
                mask = {'10': (z >= -1) & (z <= 4), '15': (z >= -0.94) & (z <= 2), '20': (z >= -0.76) & (z <= 1.4)}
                z = z[mask[str(int(args.r))]]
                lhc_y = lhc_y[:, mask[str(int(args.r))]]

        save_dir = '/pscratch/sd/m/mpinon/density/cov/'
        save_fn = Path(save_dir) / '{}_r{:.0f}_cov_lhc{}{}.npy'.format(args.stat, args.r, '_logbins' if args.logbins else '', '_cut' if args.cut else '')
        if args.stat == 'pdf':
            cout = {'delta': z, 'lhc_x': lhc_x, 'lhc_y': lhc_y, 'lhc_x_names': lhc_x_names}
        elif args.stat == 'cgf':
            cout = {'lambda': z, 'lhc_x': lhc_x, 'lhc_y': lhc_y, 'lhc_x_names': lhc_x_names}
        print('Saving compressed measurements at {}.'.format(save_fn))
        np.save(save_fn, cout)