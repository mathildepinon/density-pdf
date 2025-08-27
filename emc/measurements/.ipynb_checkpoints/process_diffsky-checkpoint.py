import numpy as np
from pathlib import Path
import pandas
import fitsio
import argparse


def read_lhc(statistic='cgf', r=10, return_sep=False, los='combine', filter_low_n=False, data_dir='/pscratch/sd/m/mpinon/density/emc', simname='galsampled_diffsky_mock_67120_fixedAmp_001_mass_conc_v0.3', logbins=False):
    lhc_y = []
    sep = []
    if los=='combine':
        data_list = list()
        for i, l in enumerate(['x', 'y', 'z']):
            data_fn = Path(data_dir) / 'density_{}_cellsize4_r{:.0f}_{}_los{}{}.npy'.format(statistic, r, simname, l, '_logbins' if logbins else '')
            print('Loading {}'.format(data_fn))
            data = np.load(data_fn, allow_pickle=True)
            data_list.append(data)
        edges = np.mean([data_list[i]['edges' if statistic=='pdf' else 'lambda'] for i in range(3)], axis=0)
        stat = np.mean([data_list[i][statistic] for i in range(3)], axis=0)
    else:
        data_fn = Path(data_dir) / 'density_{}_cellsize4_r{:.0f}_{}_los{}{}.npy'.format(statistic, r, simname, los, '_logbins' if logbins else '')
        data = np.load(data_fn, allow_pickle=True)
        edges, stat = data['edges' if statistic=='pdf' else 'lambda'], data[statistic]
    if statistic=='pdf':
        edges = (edges[1:]+edges[:-1])/2
    else:
        edges = edges

    sep = edges
    lhc_y = np.array(stat)
    if return_sep:
        return sep, lhc_y
    return lhc_y

def rebin_pdf(x, y, n=2):
    x = np.mean(x.reshape((-1, n)), axis=1)
    y = np.mean(y.reshape((-1, n)), axis=-1)
    return x, y


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--stat", type=str, default='pdf', choices=['pdf', 'cgf'])
    parser.add_argument("--basesim", type=str, default='unit', choices=['unit', 'abacus'])
    parser.add_argument("--r", type=float, default=10)
    parser.add_argument("--phase", type=int, default=1)
    parser.add_argument("--galsampled", type=str, default='mass_conc', choices=['mass_conc', 'mass'])
    parser.add_argument("--logbins", type=bool, default=False)
    args = parser.parse_args()

    data_dir = '/pscratch/sd/m/mpinon/density/emc/{}/z{:.1f}/r{:.0f}'.format(args.basesim, 0.5, args.r)
    if args.basesim=='unit':
        simname='galsampled_diffsky_mock_67120_fixedAmp_{:03d}_{}_v0.3'.format(args.phase, args.galsampled)
    z, y = read_lhc(statistic=args.stat, r=args.r, return_sep=True, data_dir=data_dir, simname=simname, logbins=args.logbins)
    save_dir = '/pscratch/sd/m/mpinon/density/diffsky/'
    save_name = '{}_r{:.0f}_{}.npy'.format(args.stat, args.r, simname)
    save_fn = Path(save_dir) / save_name
    if args.stat == 'pdf':
        cout = {'delta': z, 'pdf': y}
    elif args.stat == 'cgf':
        cout = {'lambda': z, 'cgf': y}
    print('Saving measurements at {}.'.format(save_fn))
    np.save(save_fn, cout)

    if args.stat=='cgf':
        rebin = {'10': 4, '15': 8, '20': 8}
        z = z[::rebin[str(int(args.r))]]
        y = y[::rebin[str(int(args.r))]]
        mask = {'10': (z >= -200) & (z <= -0.5), '15': (z >= -500) & (z <= -0.5), '20': (z >= -500) & (z <= -0.5)}
        z = z[mask[str(int(args.r))]]
        y = y[mask[str(int(args.r))]]
        cout['cgf'] = y
        cout['lambda'] = z
        save_name = '{}_r{:.0f}_{}{}_cut.npy'.format(args.stat, args.r, simname, '_logbins' if args.logbins else '')
        save_fn = Path(save_dir) / save_name
        print('Saving cut measurements at {}.'.format(save_fn))
        np.save(save_fn, cout)

    if args.stat=='pdf':
        z, y = rebin_pdf(z, y, n=4)
        mask = {'10': (z >= -1) & (z <= 4), '15': (z >= -0.94) & (z <= 2), '20': (z >= -0.76) & (z <= 1.4)}
        z = z[mask[str(int(args.r))]]
        y = y[mask[str(int(args.r))]]
        cout['pdf'] = y
        cout['delta'] = z
        save_name = '{}_r{:.0f}_{}_cut.npy'.format(args.stat, args.r, simname)
        save_fn = Path(save_dir) / save_name
        print('Saving cut measurements at {}.'.format(save_fn))
        np.save(save_fn, cout)


 
