import os
import numpy as np
from pathlib import Path

# CGF
for r in [10, 15, 20]:
    cov_dir = Path('/Users/mp270220/Work/emc/cov/')
    cov_fn = Path(cov_dir) / 'cgf_r{:d}_cov_lhc.npy'.format(r)
    lhc_cov_cgf = np.load(cov_fn, allow_pickle=True).item()

    data_dir = Path('/Users/mp270220/Work/emc/training_sets/cosmo+hod/ph000/seed0')
    data_fn = Path(data_dir) / 'cgf_r{:d}_lhc.npy'.format(r)
    lhc_cgf = np.load(data_fn, allow_pickle=True).item()
    
    if r == 10:
        lda_sample = np.array([-50., -20., -10., -5., -4., -3., -2., -1.5, -1., -0.5, -0.2])
    if r == 15:
        lda_sample = np.array([-50., -20, -10, -5., -4., -2., -1., -0.2])
    if r == 20:
        lda_sample = np.array([-50., -20., -10, -5, -1, -0.2])

    lda = lhc_cov_cgf['lambda']
    mask = np.isin(np.around(lda, 2), np.around(lda_sample, 2))
    print('cgf cov r = {}'.format(r), lhc_cov_cgf['lambda'][mask])

    lhc_cov_cgf_cut = lhc_cov_cgf.copy()   
    lhc_cov_cgf_cut['lhc_y'] = lhc_cov_cgf['lhc_y'][:, mask]
    lhc_cov_cgf_cut['lambda'] = lhc_cov_cgf['lambda'][mask]
    cov_fn_cut = Path(cov_dir) / 'cgf_r{:d}_cov_lhc_cut.npy'.format(r)
    np.save(cov_fn_cut, lhc_cov_cgf_cut)

    lda = lhc_cgf['lambda']
    mask = np.isin(np.around(lda, 2), np.around(lda_sample, 2))
    print('cgf data r = {}'.format(r), lhc_cgf['lambda'][mask])

    lhc_cgf_cut = lhc_cgf.copy()   
    lhc_cgf_cut['lhc_y'] = lhc_cgf['lhc_y'][:, mask]
    lhc_cgf_cut['lambda'] = lhc_cgf['lambda'][mask]
    data_fn_cut = Path(data_dir) / 'cgf_r{:d}_lhc_cut.npy'.format(r)
    np.save(data_fn_cut, lhc_cgf_cut)

def rebin(x, y):
    y = (y[:, 1:][:, ::2] + y[:, :-1][:, ::2])/2
    x = (x[1:][::2] + x[:-1][::2])/2
    return x, y

# PDF
for r in [10, 15, 20]:
    cov_dir = Path('/Users/mp270220/Work/emc/cov/')
    cov_fn = Path(cov_dir) / 'pdf_r{:d}_cov_lhc.npy'.format(r)
    lhc_cov_pdf = np.load(cov_fn, allow_pickle=True).item()

    data_dir = Path('/Users/mp270220/Work/emc/training_sets/cosmo+hod/ph000/seed0')
    data_fn = Path(data_dir) / 'pdf_r{:d}_lhc.npy'.format(r)
    lhc_pdf = np.load(data_fn, allow_pickle=True).item()
    
    x, y = rebin(lhc_cov_pdf['delta'], lhc_cov_pdf['lhc_y'])
    lhc_cov_pdf['delta'] = x
    lhc_cov_pdf['lhc_y'] = y
    delta = lhc_cov_pdf['delta']

    if r==10:
        mask = delta < 4
    elif r==15:
        mask = (delta > -0.94) & (delta < 2)
    elif r==20:
        mask = (delta > -0.76) & (delta < 1.4)
    delta = delta[mask]
    print('pdf cov r = {}'.format(r), len(delta))

    lhc_cov_pdf_cut = lhc_cov_pdf.copy()   
    lhc_cov_pdf_cut['lhc_y'] = lhc_cov_pdf['lhc_y'][:, mask]
    lhc_cov_pdf_cut['delta'] = lhc_cov_pdf['delta'][mask]
    cov_fn_cut = Path(cov_dir) / 'pdf_r{:d}_cov_lhc_cut.npy'.format(r)
    np.save(cov_fn_cut, lhc_cov_pdf_cut)

    x, y = rebin(lhc_pdf['delta'], lhc_pdf['lhc_y'])
    lhc_pdf['delta'] = x
    lhc_pdf['lhc_y'] = y
 
    lhc_pdf_cut = lhc_pdf.copy()   
    lhc_pdf_cut['lhc_y'] = lhc_pdf['lhc_y'][:, mask]
    lhc_pdf_cut['delta'] = lhc_pdf['delta'][mask]
    data_fn_cut = Path(data_dir) / 'pdf_r{:d}_lhc_cut.npy'.format(r)
    np.save(data_fn_cut, lhc_pdf_cut)