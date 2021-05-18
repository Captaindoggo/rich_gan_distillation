import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

import torch.utils.data as data


def get_scaler(n_quantiles):
    if n_quantiles > 0:
        return QuantileTransformer(n_quantiles=n_quantiles, output_distribution='normal', subsample=int(1e10))
    else:
        return StandardScaler()


class DataHandler:
    def __init__(self, config):
        self.config = config
        self.scalers = {
            'data': get_scaler(config.data.scaler.n_quantiles),
            'context': get_scaler(config.data.scaler.n_quantiles),
            'weight': get_scaler(config.experiment.weights.n_quantiles),
        }

        if self.config.experiment.weights.positive:
            self.scalers['weight'] = MinMaxScaler()
        if not self.config.experiment.weights.enable:
            self.scalers['weight'] = NoneProcessor()

        if config.data.download:
            if not os.path.exists(config.data.data_path):
                os.makedirs(config.data.data_path)
            log.info('config.data.download is True, starting dowload')
            target_path = os.path.join(config.data.data_path, 'data-calibsample')
            if os.path.exists(target_path):
                print("It seems that data is already downloaded. Are you sure?")
            os.system(f"wget https://cernbox.cern.ch/index.php/s/Fjf3UNgvlRVa4Td/download -O {target_path + '.tar.gz'}")
            log.info('files downloaded, starting unpacking')
            os.system(f"tar xvf {target_path + '.tar.gz'}")
            log.info('files unpacked')

        # todo rethink
        config.data.data_path = os.path.join(config.data.data_path, 'data-calibsample')

        table = np.array(get_particle_table(config.data.data_path, config.experiment.particle))
        train_table, val_table = train_test_split(table, test_size=self.config.data.val_size, random_state=42)
        self.scalers['data'].fit(train_table[:, :config.experiment.data.data_dim])
        self.scalers['context'].fit(
            train_table[:, config.experiment.data.data_dim:
                           config.experiment.data.data_dim + config.experiment.data.context_dim]
        )
        self.scalers['weight'].fit(train_table[:, -1].reshape(-1, 1))
        # todo assert weight on last col, mb add to config

        train_table = np.concatenate([
            self.scalers['data'].transform(train_table[:, :config.experiment.data.data_dim]),
            self.scalers['context'].transform(
                train_table[:, config.experiment.data.data_dim:
                               config.experiment.data.data_dim + config.experiment.data.context_dim]),
            self.scalers['weight'].transform(train_table[:, -1].reshape(-1, 1))
        ], axis=1)
        val_table = np.concatenate([
            self.scalers['data'].transform(val_table[:, :config.experiment.data.data_dim]),
            self.scalers['context'].transform(
                val_table[:, config.experiment.data.data_dim:
                             config.experiment.data.data_dim + config.experiment.data.context_dim]),
            self.scalers['weight'].transform(val_table[:, -1].reshape(-1, 1))
        ], axis=1)

        train_dataset = ParticleDataset(config, train_table)
        self.train_loader = data.DataLoader(
            dataset=train_dataset,
            batch_size=config.experiment.batch_size,
            sampler=data.DistributedSampler(train_dataset) if config.utils.use_ddp else None,
            shuffle=True if not config.utils.use_ddp else None,
            pin_memory=True,
            drop_last=True
        )
        val_dataset = ParticleDataset(config, val_table)
        self.val_loader = data.DataLoader(
            dataset=val_dataset,
            batch_size=config.experiment.batch_size,
            sampler=None,
            shuffle=False,
            drop_last=True
        )


class ParticleDataset(torch.utils.data.Dataset):
    def __init__(self, config, table):
        self.data = table[:, :config.experiment.data.data_dim]
        self.context = table[:, config.experiment.data.data_dim:
                                     config.experiment.data.data_dim + config.experiment.data.context_dim]
        self.weight = table[:, -1]
        assert config.experiment.data.data_dim + config.experiment.data.context_dim + 1 == table.shape[1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.context[idx], self.weight[idx]


dll_columns = ['RichDLLe', 'RichDLLk', 'RichDLLmu', 'RichDLLp', 'RichDLLbt']
raw_feature_columns = ['Brunel_P', 'Brunel_ETA', 'nTracks_Brunel']
weight_col = 'probe_sWeight'
list_particles = ['kaon', 'pion', 'proton', 'muon', 'electron']


def load_and_cut(file_name):
    data = pd.read_csv(file_name, delimiter='\t')
    return data[dll_columns + raw_feature_columns + [weight_col]]


def load_and_merge_and_cut(filename_list):
    return pd.concat([load_and_cut(fname) for fname in filename_list], axis=0, ignore_index=True)


def get_particle_table(data_path, particle):
    particle_files = [os.path.join(data_path, name) for name in os.listdir(data_path) if particle in name]
    particle_csv = []
    for path in particle_files:
        table = pd.read_csv(path, delimiter='\t')
        table = table[dll_columns + raw_feature_columns + [weight_col]]
        particle_csv.append(table)
    particle_csv = pd.concat(particle_csv, axis=0, ignore_index=True)
    return particle_csv


class NoneProcessor(StandardScaler):
    def fit(self, X, y=None):
        pass

    def transform(self, X, copy=None):
        return X

    def inverse_transform(self, X, copy=None):
        return X


class DotDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, dct):
        for key, value in dct.items():
            if hasattr(value, 'keys'):
                value = DotDict(value)
            self[key] = value

    def _getattr(self, key):
        target = self
        for dot in key.split('.'):
            target = target[dot]
        return target

    def _setattr(self, key, value):
        target = self
        for dot in key.split('.')[:-1]:
            target = target[dot]
        target[key.split('.')[-1]] = value