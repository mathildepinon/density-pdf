import time
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sunbird.emulators import FCN, train
from sunbird.data import ArrayDataModule
import torch

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()

    parser = argparse.ArgumentParser()
    parser.add_argument("--stat", type=str, default='pdf', choices=['pdf', 'cgf'])
    parser.add_argument("--variation", type=str, default='hod', choices=['hod', 'cosmo+hod'])
    parser.add_argument("--r", type=float, default=10)
    parser.add_argument("--test_frac", type=float, default=0.1)

    args = parser.parse_args()

    if args.variation == 'hod':
        data_dir = Path('/pscratch/sd/m/mpinon/density/training_sets/hod/c000_ph000/seed0/')
    elif args.variation == 'cosmo+hod':
        data_dir = Path('/pscratch/sd/m/mpinon/density/training_sets/cosmo+hod/ph000/seed0/')
    data_fn = Path(data_dir) / '{}_r{:.0f}{}_lhc.npy'.format(args.stat, args.r, '_logbins' if args.stat=='cgf' else '')
    lhc = np.load(data_fn, allow_pickle=True).item()
    lhc_x = lhc['lhc_x']
    lhc_x_names = lhc['lhc_x_names']
    lhc_y = lhc['lhc_y']

    if args.stat=='cgf':
        # truncate CGF to avoid infinite values
        mask = lhc['lambda'] < 7
        lhc_y = lhc_y[:, mask]

    n = len(lhc_y)
    n_val = int(np.round(args.test_frac * n))
    print("Total data size is {}. Keeping {} data points for training and {} for testing.".format(n, n-n_val, n_val))

    lhc_train_x = lhc_x[:-n_val]
    lhc_train_y = lhc_y[:-n_val]
    lhc_test_x = lhc_x[-n_val:]
    lhc_test_y = lhc_y[-n_val:]

    train_mean = np.mean(lhc_y, axis=0)
    train_std = np.std(lhc_y, axis=0)

    train_mean_x = np.mean(lhc_x, axis=0)
    train_std_x = np.std(lhc_x, axis=0)

    print('nans: ', np.all(np.isfinite(lhc_train_y)))

    data = ArrayDataModule(x=torch.Tensor(lhc_train_x),
                        y=torch.Tensor(lhc_train_y), 
                        val_fraction=0.2, batch_size=256,
                        num_workers=2)
    data.setup()

    model = FCN(
            n_input=data.n_input,
            n_output=data.n_output,
            n_hidden=[512, 512, 512, 512],
            dropout_rate=0, 
            learning_rate=1.e-3,
            scheduler_patience=30,
            scheduler_factor=0.5,
            scheduler_threshold=1.e-6,
            weight_decay=0.,
            act_fn='learned_sigmoid',
            #act_fn='SiLU',
            loss='rmse',
            training=True,
            mean_output=train_mean,
            std_output=train_std,
            mean_input=train_mean_x,
            std_input=train_std_x
        )

    val_loss, model, early_stop_callback = train.fit(
        data=data, model=model,
        model_dir='/global/homes/m/mpinon/density/emulator/{}/r{:.0f}/{}/'.format(args.stat, args.r, args.variation),
    )

    print('best score = ', early_stop_callback.best_score.item())
    print('stopped epoch = ', early_stop_callback.stopped_epoch)
    print('wait count = ', early_stop_callback.wait_count)