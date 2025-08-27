import time
import optuna
import joblib
from pathlib import Path
import argparse
import numpy as np

from sunbird.emulators import FCN, train
print(repr(train))
from sunbird.data import ArrayDataModule
from sunbird.data.data_utils import convert_to_summary
from sunbird.data.transforms_array import LogTransform, ArcsinhTransform
import torch


data_dir = Path('/pscratch/sd/m/mpinon/density/training_sets/cosmo+hod/ph000/seed0/')
cov_dir = Path('/pscratch/sd/m/mpinon/density/cov/')


def TrainFCN(stat, r, learning_rate, n_hidden, dropout_rate, weight_decay, 
    batch_size, model_dir=None, transform=False):
    #observable = getattr(emc, observable)()
    data_fn = Path(data_dir) / '{}_r{:.0f}_lhc{}_cut.npy'.format(stat, r, '_logbins' if stat=='cgf' else '')
    lhc = np.load(data_fn, allow_pickle=True).item()
    cov_fn = Path(cov_dir) / '{}_r{:.0f}_cov_lhc{}_cut.npy'.format(stat, r, '_logbins' if stat=='cgf' else '')
    lhc_cov = np.load(cov_fn, allow_pickle=True).item()

    final_model = False
    apply_transform = transform
    select_filters = {}
    slice_filters = {}

    # load the data
    lhc_x = lhc['lhc_x']
    lhc_y = lhc['lhc_y']
    covariance_matrix = np.cov(lhc_cov['lhc_y'], rowvar=False) / 64
    coordinates = {'lambda': lhc['lambda']} if stat=='cgf' else {'delta': lhc['delta']}
    print(f'Loaded LHC with shape: {lhc_x.shape}, {lhc_y.shape}')

    # reshape the features to have the format (n_samples, n_features)
    cosmo_idx = list(range(0, 5)) + list(range(13, 14)) + list(range(100, 127)) + list(range(130, 182))
    hod_idx = list(range(350))
    n_samples = len(cosmo_idx) * len(hod_idx)
    lhc_x = lhc_x.reshape(n_samples, -1)
    lhc_y = lhc_y.reshape(n_samples, -1)
    print(f'Reshaped LHC with shape: {lhc_x.shape}, {lhc_y.shape}')

    # covariance_matrix = observable.get_covariance_matrix(divide_factor=64)
    # print(f'Loaded covariance matrix with shape: {covariance_matrix.shape}')

    if apply_transform:
        if apply_transform=='asinh':
            transform = ArcsinhTransform()
        elif apply_transform=='log':
            transform = LogTransform()
        lhc_y = transform.transform(lhc_y)
    else:
        transform = None
        

    nhod = int(len(lhc_y) / 85)
    ntot = len(lhc_y)

    if final_model:
        idx_train = list(range(ntot))
    else:
        idx_train = list(range(nhod * 6, ntot))


    print(f'Using {len(idx_train)} samples for training')

    lhc_train_x = lhc_x[idx_train]
    lhc_train_y = lhc_y[idx_train]

    train_mean = np.mean(lhc_train_y, axis=0)
    train_std = np.std(lhc_train_y, axis=0)

    train_mean_x = np.mean(lhc_train_x, axis=0)
    train_std_x = np.std(lhc_train_x, axis=0)

    data = ArrayDataModule(x=torch.Tensor(lhc_train_x),
                        y=torch.Tensor(lhc_train_y), 
                        val_fraction=0.1, batch_size=batch_size,
                        num_workers=0)
    data.setup()

    model = FCN(
            n_input=data.n_input,
            n_output=data.n_output,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate, 
            learning_rate=learning_rate,
            scheduler_patience=10,
            scheduler_factor=0.5,
            scheduler_threshold=1.e-6,
            weight_decay=weight_decay,
            act_fn='learned_sigmoid',
            # act_fn='SiLU',
            # loss='GaussianNLoglike',
            loss='mae',
            training=True,
            mean_output=train_mean,
            std_output=train_std,
            mean_input=train_mean_x,
            std_input=train_std_x,
            transform_output=transform,
            standarize_output=True,
            coordinates=coordinates,
            covariance_matrix=covariance_matrix,
        )

    model_dir = './' if model_dir is None else model_dir
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    print(f'Saving model to {model_dir}')

    val_loss, model, early_stop_callback = train.fit(
        data=data, model=model,
        model_dir=model_dir,
        max_epochs=5000,
        devices=1,
    )
    return val_loss


def objective(
    trial,
):
    nlayers = 4
    n_hidden = [512] * nlayers
    learning_rate = trial.suggest_float(
        "learning_rate",
        1.0e-3,
        0.01,
    )
    weight_decay = trial.suggest_float("weight_decay", 1.0e-5, 0.001)
    # same_n_hidden = True
    # n_layers = trial.suggest_int("n_layers", 1, 10)
    # if same_n_hidden:
    #     n_hidden = [trial.suggest_int("n_hidden", 200, 1024)] * n_layers
    # else:
    #     n_hidden = [
    #         trial.suggest_int(f"n_hidden_{layer}", 200, 1024)
    #         for layer in range(n_layers)
    #     ]
    dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.15)
    batch_size = trial.suggest_int("batch_size", 64, 512)
    return TrainFCN(learning_rate=learning_rate, n_hidden=n_hidden,
                    dropout_rate=dropout_rate, weight_decay=weight_decay,
                    batch_size=batch_size, stat=args.stat, r=args.r,
                    model_dir=study_dir, transform=transform)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--stat", type=str, default='pdf', choices=['pdf', 'cgf'])
    parser.add_argument("--r", type=float, default=10)
    parser.add_argument("--transform", type=str, default='asinh', choices=['', 'asinh', 'log'])
    args = parser.parse_args()
    stat = args.stat

    n_trials = 10
    transform = args.transform
    study_dir = '/pscratch/sd/m/mpinon/density/trained_models/{}/r{:.0f}/cosmo+hod/optuna/{}'.format(stat, args.r, transform)
    Path(study_dir).mkdir(parents=True, exist_ok=True)
    study_fn = Path(study_dir) / f'{stat}.pkl'
    print('study_fn', study_fn)

    t0 = time.time()
    tmp = t0

    for i in range(n_trials):
        if study_fn.exists():
            print(f"Loading existing study from {study_fn}")
            study = joblib.load(study_fn)
        else:
            study = optuna.create_study(study_name=f'{stat}')
        optimize_objective = lambda trial: objective(trial)
        study.optimize(optimize_objective, n_trials=1)

        print("Best trial:")
        trial = study.best_trial
        print("  Value: {}".format(trial.value))
        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))
        joblib.dump(
            study,
            study_fn,
        )

        tmp = time.time() - tmp
        print('Trial {} done in elapsed time {}s'.format(i, tmp))

    t1 = time.time()
    print('Optimization done in elapsed time {}s'.format(t1 - t0))