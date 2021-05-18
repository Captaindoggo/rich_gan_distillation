import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.nn.utils import spectral_norm


class GLayer(nn.Module):
    def __init__(self, indim, outdim):
        super(GLayer, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(indim, outdim),
            nn.BatchNorm1d(outdim),
            nn.ReLU()
        )
    def forward(self, x):
        x = self.block(x)
        return x

class CLayer(nn.Module):
    def __init__(self, indim, outdim):
        super(CLayer, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(indim, outdim),
            nn.LayerNorm(normalized_shape=outdim),
            nn.ReLU()
        )
    def forward(self, x):
        x = self.block(x)
        return x


class Generator(nn.Module):
    def __init__(self, config, nlayers):
        super(Generator, self).__init__()
        self.config = config
        dimslist = [[self.config.model.G.noise_dim + self.config.experiment.data.context_dim, 128]]
        dimslist.extend([[128, 128] for i in range(nlayers - 2)])

        self.layers = nn.ModuleList([GLayer(dim[0], dim[1]) for dim in dimslist])
        self.out = nn.Linear(128, self.config.experiment.data.data_dim)

    def _sample_latent(self, x):
        return torch.randn(x.shape[0], self.config.model.G.noise_dim, device=self.config.utils.device)

    def forward(self, noise, context):
        # noise = self._sample_latent(x)
        # todo add reduce_noise
        x = torch.cat([noise.float(), context.float()], dim=1)
        for layer in self.layers:
            x = layer(x)
        x = self.out(x)
        return x

    def get_activations(self, noise, context, n_acts):
        act_list = []
        x = torch.cat([noise.float(), context.float()], dim=1)
        n = 0
        for layer in self.layers:
            x = layer(x)
            if n in set(n_acts):
                act_list.append(x)
            n += 1
        if -1 in set(n_acts) or n in set(n_acts):
            x = self.out(x)
            act_list.append(x)
        return act_list


class Critic(nn.Module):
    def __init__(self, config, nlayers):
        super(Critic, self).__init__()
        self.config = config

        dimslist = [[self.config.experiment.data.data_dim + self.config.experiment.data.context_dim, 128]]
        dimslist.extend([[128, 128] for i in range(nlayers - 2)])

        self.layers = nn.ModuleList([CLayer(dim[0], dim[1]) for dim in dimslist])
        self.out = nn.Linear(128, 256)

    def forward(self, x, context):
        x = torch.cat([x.float(), context.float()], dim=1)
        for layer in self.layers:
            x = layer(x)
        x = self.out(x)
        return x

    def get_activations(self, x, context, n_acts):
        act_list = []
        x = torch.cat([x.float(), context.float()], dim=1)
        n = 0
        for layer in self.layers:
            x = layer(x)
            if n in set(n_acts):
                act_list.append(x)
            n += 1
        if -1 in set(n_acts) or n in set(n_acts):
            x = self.out(x)
            act_list.append(x)
        return act_list


class RICHGAN(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.G = Generator(config, config.model.G.num_layers)
        self.C = Critic(config, config, config.model.C.num_layers)

        self.optim_g = optim.Adam(self.G.parameters(), lr=config.experiment.lr.G, betas=(0.5, 0.9))
        self.optim_c = optim.Adam(self.C.parameters(), lr=config.experiment.lr.C, betas=(0.5, 0.9))

    @torch.no_grad()
    def generate(self, noise, context):
        return self.G(noise, context)

    def save(self, path):
        states = {
            'G': self.G.state_dict(),
            'C': self.C.state_dict(),
            'G_optim': self.optim_g.state_dict(),
            'C_optim': self.optim_c.state_dict(),
        }
        torch.save(states, path)

    def load(self, path):
        states = torch.load(path, map_location=lambda storage, loc: storage)
        if 'G' in states:
            self.G.load_state_dict(states['G'])
        if 'C' in states:
            self.C.load_state_dict(states['C'])
        if 'G_optim' in states:
            self.optim_g.load_state_dict(states['G_optim'])
        if 'C_optim' in states:
            self.optim_c.load_state_dict(states['C_optim'])




class StudentRICHGAN(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.G = Generator(config, config.student.G.num_layers)
        self.C = Critic(config, config, config.student.C.num_layers)

        self.optim_g = optim.Adam(self.G.parameters(), lr=config.experiment.lr.G, betas=(0.5, 0.9))
        self.optim_c = optim.Adam(self.C.parameters(), lr=config.experiment.lr.C, betas=(0.5, 0.9))

    @torch.no_grad()
    def generate(self, noise, context):
        return self.G(noise, context)

    def save(self, path):
        states = {
            'G': self.G.state_dict(),
            'C': self.C.state_dict(),
            'G_optim': self.optim_g.state_dict(),
            'C_optim': self.optim_c.state_dict(),
        }
        torch.save(states, path)

    def load(self, path):
        states = torch.load(path, map_location=lambda storage, loc: storage)
        if 'G' in states:
            self.G.load_state_dict(states['G'])
        if 'C' in states:
            self.C.load_state_dict(states['C'])
        if 'G_optim' in states:
            self.optim_g.load_state_dict(states['G_optim'])
        if 'C_optim' in states:
            self.optim_c.load_state_dict(states['C_optim'])