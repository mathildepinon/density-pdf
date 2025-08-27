import os
import argparse
import numpy as np
from pathlib import Path
from sunbird.inference.samples import Chain
import matplotlib.pyplot as plt
from getdist import plots, MCSamples

plt.style.use('/global/homes/m/mpinon/density/plot_style.mplstyle')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--stat", type=str, nargs='+', default=['pdf', 'pk_kmax0.2', 'pk_kmax0.2+pdf'], choices=['pdf', 'cgf', 'pk_kmax0.2', 'pk_kmax0.5', 'pk_kmax0.2+pdf', 'pk_kmax0.5+pdf', 'pk_kmax0.2+cgf', 'dsc_pk_kmax0.2', 'pk_kmax0.2+dsc_pk_kmax0.2'])
    parser.add_argument("--params", type=str, default='cosmo', choices=['cosmo', 'hod', 'hod_base', 'hod_extended', 'all'])

    args = parser.parse_args()
    stats = args.stat

    # params to show
    if args.params == 'cosmo':
        params = ['omega_b', 'omega_cdm', 'sigma8_m', 'n_s']
    elif args.params == 'hod':
        params = ['logM_cut', 'logM_1', 'sigma', 'alpha', 'kappa', 'alpha_c', 'alpha_s', 's', 'B_cen', 'B_sat']
    elif args.params == 'hod_base':
         params = ['logM_cut', 'logM_1', 'sigma', 'alpha', 'kappa']
    elif args.params == 'hod_extended':
        params = ['alpha_c', 'alpha_s', 's', 'B_cen', 'B_sat']
    elif args.params == 'all':
        params = ['omega_b', 'omega_cdm', 'sigma8_m', 'n_s', 'logM_cut', 'logM_1', 'sigma', 'alpha', 'kappa', 'alpha_c', 'alpha_s', 's', 'B_cen', 'B_sat']

    chain_dir = '/global/cfs/cdirs/desi/users/mpinon/acm/fits_emc/abacus/c000_hod030/LCDM/'

    chains = []
    legend_labels = []
    stat_labels = {'pdf': r'PDF ($R = 10, 15, 20\,h^{-1}\mathrm{Mpc}$)', 
                    'cgf': r'CGF ($R = 10, 15, 20\,h^{-1}\mathrm{Mpc}$)', 
                    'pk_kmax0.2': r'$P_{{\ell}}(k) \; (k_\mathrm{{max}} = 0.2\,h\mathrm{{Mpc}}^{{-1}})$',
                    'pk_kmax0.5': r'$P_{{\ell}}(k) \; (k_\mathrm{{max}} = 0.5\,h\mathrm{{Mpc}}^{{-1}})$',
                    'pk_kmax0.2+pdf': r'$P_{{\ell}}(k) + \mathrm{PDF}$',
                    'pk_kmax0.5+pdf': r'$P_{{\ell}}(k) + \mathrm{PDF}$',
                    'pk_kmax0.2+cgf': r'$P_{{\ell}}(k) + \mathrm{CGF}$',
                    'dsc_pk_kmax0.2': r'$P^{\rm DS}_{\ell}(k) \; (k_\mathrm{max} = 0.2\,h\mathrm{Mpc}^{-1}))$',
                    'pk_kmax0.2+dsc_pk_kmax0.2': r'$P_{{\ell}}(k) + P^{\rm DS}_{\ell}(k) \; (k_\mathrm{max} = 0.2\,h\mathrm{Mpc}^{-1}))$'}
    
    for stat in stats:
        stat_name = 'number_density+{}'.format(stat)
        chain_fn = Path(chain_dir) / f'chain_{stat_name}.npy'
        chain = Chain.load(chain_fn)
        samples = Chain.to_getdist(chain, add_derived=False)
        chains.append(samples)
        legend_labels.append(stat_labels[stat])
    
    markers = chain.markers

    g = plots.get_single_plotter(width_inch=6, scaling=False, ratio=1)
    g.settings.axis_marker_lw = 1.0
    g.settings.axis_marker_ls = "--"
    g.settings.title_limit_labels = False
    g.settings.axis_marker_color = "k"
    g.settings.legend_colored_text = True
    g.settings.figure_legend_frame = False
    g.settings.figure_legend_ncol = 1
    g.settings.linewidth_contour = 1.0
    g.settings.legend_fontsize = 12
    g.settings.axes_fontsize = 12
    g.settings.axes_labelsize = 12
    g.settings.axis_tick_x_rotation = 45
    g.settings.solid_colors = ['#2377e1', '#e03423', '#808080']
    g.settings.line_styles = g.settings.solid_colors

    g.triangle_plot(chains, params=params, markers=markers, filled=True, 
                    legend_labels=legend_labels, legend_loc='upper right', show=False)
    fig = plt.gcf()
    fig.align_xlabels()
    fig.align_ylabels()

    stats_name = '_'.join(stats)
    plt.savefig('/global/homes/m/mpinon/density/plots/posteriors_{}_{}.pdf'.format(stats_name, args.params), dpi=300)