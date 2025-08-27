from sunbird.inference.pocomc import PocoMCSampler
from sunbird.inference import priors as sunbird_priors
from sunbird import setup_logging

import acm.observables.emc as emc

from pathlib import Path
import numpy as np
import argparse


def get_priors(cosmo=True, hod=True):
    stats_module = 'scipy.stats'
    priors, ranges, labels = {}, {}, {}
    if cosmo:
        priors.update(sunbird_priors.AbacusSummit(stats_module).priors)
        ranges.update(sunbird_priors.AbacusSummit(stats_module).ranges)
        labels.update(sunbird_priors.AbacusSummit(stats_module).labels)
    if hod:
        priors.update(sunbird_priors.Yuan23(stats_module).priors)
        ranges.update(sunbird_priors.Yuan23(stats_module).ranges)
        labels.update(sunbird_priors.Yuan23(stats_module).labels)
    return priors, ranges, labels


parser = argparse.ArgumentParser()
parser.add_argument("--cosmo_idx", type=int, default=0)
parser.add_argument("--hod_idx", type=int, default=30)
parser.add_argument("--stat", type=str, nargs='+', default=['pk', 'pdf'], choices=['pk', 'pdf', 'cgf', 'ds'])
parser.add_argument("--kmax", type=float, default=0.2)
parser.add_argument("--r", type=int, nargs='+', required=False, default=[10, 15, 20], choices=[10, 15, 20])

args = parser.parse_args()
setup_logging()

# set up the inference
priors, ranges, labels = get_priors(cosmo=True, hod=True)
# fixed_params = []
fixed_params = ['w0_fld', 'wa_fld', 'nrun', 'N_ur', 'A_cen', 'A_sat']
# , 'sigma', 'kappa', 'alpha', 's', 'A_cen', 'A_sat', 'B_cen', 'B_sat', 'alpha_s', 'alpha_c']
add_emulator_error = True
select_mocks = {'cosmo_idx': args.cosmo_idx, 'hod_idx': args.hod_idx}

# load observables with their custom filters
# observables = [emc.GalaxyOverdensityPDF(
#         r = int(args.r[i]),
#         select_mocks=select_mocks
#     )
#     for i in range(len(args.r))]

# observable = emc.CombinedObservable(observables)

observables = [
        emc.GalaxyNumberDensity(
            select_mocks=select_mocks
        )
]
if 'pk' in args.stat:
    observables.append(
        emc.GalaxyPowerSpectrumMultipoles(
            select_mocks=select_mocks,
            slice_coordinates={'k': [0, args.kmax]},
        )
    )
if 'pdf' in args.stat:
    observables.append(
        emc.GalaxyOverdensityPDF(
            select_mocks=select_mocks
        )
    )
if 'cgf' in args.stat:
    observables.append(
        emc.CumulantGeneratingFunction(
            select_mocks=select_mocks
        )
    )
if 'ds' in args.stat:
    observables.append(
        emc.DensitySplitPowerSpectrumMultipoles(
            select_mocks=select_mocks,
            slice_coordinates={'k': [0, args.kmax]},
            select_coordinates={
                'statistics': ['quantile_data_power'],
            },
        )
    )

observable = emc.CombinedObservable(observables)

statistics = observable.stat_name
if 'pk' in args.stat:
    statistics[np.argwhere(np.array(args.stat)=='pk')] += '_kmax{:.1f}'.format(args.kmax)
if 'ds' in args.stat:
    statistics[np.argwhere(np.array(args.stat)=='ds')] += '_kmax{:.1f}'.format(args.kmax)
print(f'Fitting {statistics} with cosmo_idx={args.cosmo_idx} and hod_idx={args.hod_idx}')

# load the data
data_x = observable.lhc_x
data_x_names = observable.lhc_x_names
data_y = observable.lhc_y
print(f'Loaded LHC x with shape: {data_x.shape}')
print(f'Loaded LHC y with shape {data_y.shape}')

# load the covariance matrix
covariance_matrix = observable.get_covariance_matrix(divide_factor=64)
print(f'Loaded covariance matrix with shape: {covariance_matrix.shape}')

# load emulator error
if add_emulator_error:
    emulator_error = observable.get_emulator_error()
    covariance_matrix += np.diag(emulator_error**2)

# get the debiased inverse
correction = observable.get_covariance_correction(
    n_s=len(observable.small_box_y),
    n_d=len(covariance_matrix),
    n_theta=len(data_x_names) - len(fixed_params),
    method='percival',
)
precision_matrix = np.linalg.inv(correction * covariance_matrix)

fixed_params = {key: data_x[data_x_names.index(key)]
                    for key in fixed_params}

# load the model
model = observable.model

# sample the posterior
sampler = PocoMCSampler(
    observation=data_y,
    precision_matrix=precision_matrix,
    theory_model=model,
    fixed_parameters=fixed_params,
    priors=priors,
    ranges=ranges,
    labels=labels,
    #ellipsoid=True,
)

sampler(vectorize=True, n_total=4096)
# sampler(vectorize=True, n_total=10_000)

# plot and save results
markers = {key: data_x[data_x_names.index(key)] for key in data_x_names if key not in fixed_params}
statistics = '+'.join(statistics)

save_dir = '/global/cfs/cdirs/desi/users/mpinon/acm/fits_emc/abacus/'
save_dir = Path(save_dir) / f'c{args.cosmo_idx:03}_hod{args.hod_idx:03}/LCDM/'
Path(save_dir).mkdir(parents=True, exist_ok=True)

sampler.plot_triangle(save_fn=save_dir / f'chain_{statistics}_triangle.pdf', thin=128,
                      markers=markers, title_limit=1)
sampler.plot_trace(save_fn=save_dir / f'chain_{statistics}_trace.pdf', thin=128)
sampler.save_chain(save_fn=save_dir / f'chain_{statistics}.npy', metadata={'markers': markers, 'zeff': 0.5})
sampler.save_table(save_fn=save_dir / f'chain_{statistics}_stats.txt')
sampler.plot_bestfit(save_fn=save_dir / f'chain_{statistics}_bestfit.png', model='maxl')
sampler.plot_bestfit(save_fn=save_dir / f'chain_{statistics}_mean.png', model='mean')