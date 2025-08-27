import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import matplotlib.pyplot as plt
from sunbird.emulators import FCN, train
from sunbird.data import ArrayDataModule

plt.style.use('/global/homes/m/mpinon/density/plot_style.mplstyle')


def get_best_emulator_version(r=10):
    em_err_list = list()
    if r==10:
        versions = np.arange(20) 
    elif r==15:
        versions = np.arange(8)
    else:
        versions = np.arange(18)

    for v in versions:
        if v==0:
            checkpoint_fn = "/pscratch/sd/m/mpinon/density/trained_models/pdf/r{:d}/{}/optuna/{}/last.ckpt".format(r, variation, transform)
        else:
            checkpoint_fn = "/pscratch/sd/m/mpinon/density/trained_models/pdf/r{:d}/{}/optuna/{}/last-v{:d}.ckpt".format(r, variation, transform, v)
        model = FCN.load_from_checkpoint(checkpoint_fn, strict=True)
        model = model.to('cpu')

        # and now let's test it
        model.eval()
        with torch.no_grad():
            pred_test_y  = model.get_prediction(torch.Tensor(lhc_test_x))
        pred_test_y = pred_test_y.numpy()
        pred_test_y = pred_test_y

        emulator_error = (pred_test_y - lhc_test_y)/std_pdf
        error_median = np.median(np.abs(emulator_error), axis=0)
        em_err_list.append(error_median)

    best = versions[np.argmin(np.mean(np.array(em_err_list), axis=1))]
    print(best)
    return best


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--stat", type=str, default='pdf', choices=['pdf', 'cgf'])
    parser.add_argument("--variation", type=str, default='cosmo+hod', choices=['hod', 'cosmo+hod'])
    parser.add_argument("--test_frac", type=float, default=0.1)

    args = parser.parse_args()
    variation = args.variation
    stat = args.stat

    #fig, ax = plt.subplots(2, 1, figsize=(4, 4), sharex=True, height_ratios=(4, 1))
    plt.figure(figsize=(4, 3))

    for ridx, r in enumerate([10, 15, 20]):
        if variation == 'hod':
            data_dir = Path('/pscratch/sd/m/mpinon/density/training_sets/hod/c000_ph000/seed0/')
        elif variation == 'cosmo+hod':
            data_dir = Path('/pscratch/sd/m/mpinon/density/training_sets/cosmo+hod/ph000/seed0/')
        data_fn = Path(data_dir) / '{}_r{:d}_lhc_cut.npy'.format(stat, r)
        lhc = np.load(data_fn, allow_pickle=True,).item()

        delta = lhc['delta']
        lhc_x = lhc['lhc_x']
        lhc_x_names = lhc['lhc_x_names']
        lhc_y = lhc['lhc_y']

        nhod = int(len(lhc_y) / 85)
        ntot = len(lhc_y)
        idx_train = list(range(nhod * 6, ntot))
        idx_test = list(range(0, nhod * 6))

        lhc_train_x = lhc_x[idx_train]
        lhc_train_y = lhc_y[idx_train]
        lhc_test_x = lhc_x[idx_test]
        lhc_test_y = lhc_y[idx_test]   

        # Compare to cosmic variance

        # rescale cov to the volume of the abacus mocks
        prefactor = 1 / 64

        # Read small boxes measurements for covariance
        data_dir = Path('/pscratch/sd/m/mpinon/density/cov/')
        data_fn = Path(data_dir) / '{}_r{:d}_cov_lhc_cut.npy'.format(stat, r)
        lhc_cov_pdf = np.load(data_fn, allow_pickle=True,).item()
        cov_pdf = prefactor * np.cov(lhc_cov_pdf['lhc_y'], rowvar=False)
        mean_pdf = np.mean(lhc_cov_pdf['lhc_y'], axis=0)
        std_pdf = np.diag(cov_pdf)**0.5

        transform = 'asinh'
        best = get_best_emulator_version(r=r)
        if best==0:
            checkpoint_fn = "/pscratch/sd/m/mpinon/density/trained_models/{}/r{:d}/{}/optuna/{}/last.ckpt".format(stat, r, variation, transform)
        else:
            checkpoint_fn = "/pscratch/sd/m/mpinon/density/trained_models/{}/r{:d}/{}/optuna/{}/last-v{:d}.ckpt".format(stat, r, variation, transform, best)
        model = FCN.load_from_checkpoint(checkpoint_fn, strict=True)
        model = model.to('cpu')

        # and now let's test it
        model.eval()
        with torch.no_grad():
            pred_test_y  = model.get_prediction(torch.Tensor(lhc_test_x))
        pred_test_y = pred_test_y.numpy()
        pred_test_y = pred_test_y

        i = 1
        # ax[0].plot(delta, lhc_test_y.T[:, i], color='C{}'.format(ridx), alpha=0.5, label='mock' if ridx==0 else '')
        # ax[0].plot(delta, pred_test_y.T[:, i], color='C{}'.format(ridx), ls='--', label='prediction' if ridx==0 else '')
        # ax[1].plot(delta, (lhc_test_y.T[:, i]-pred_test_y.T[:, i])/std_pdf, color='C{}'.format(ridx), ls='--')
        plt.plot(delta, lhc_test_y.T[:, i], color='C{}'.format(ridx), alpha=0.5, label='mock' if ridx==0 else '')
        plt.plot(delta, pred_test_y.T[:, i], color='C{}'.format(ridx), ls='--', label='prediction' if ridx==0 else '')
        
    if stat=='pdf':
        #ax[1].set_xlabel(r'$\delta_R$')
        plt.xlabel(r'$\delta_R$')
    plt.ylabel(r'$\mathcal{P}(\delta_R)$')
    #ax[0].set_ylabel(r'$\mathcal{P}(\delta_R)$')
    #ax[1].set_ylabel(r'$(\mathcal{P}_{\mathrm{pred}} - \mathcal{P}_{\mathrm{test}})/\sigma$')
    #ax[0].legend()
    plt.legend()
    #plt.ylim(ymax=2)
    plt.gcf().align_ylabels()
    plt.savefig('/global/homes/m/mpinon/density/plots/emulator_prediction_test_{}.pdf'.format(stat), dpi=300)