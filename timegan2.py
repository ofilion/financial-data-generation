"""
PyTorch implementation of TimeGAN based on https://github.com/ydataai/ydata-synthetic/blob/dev/src/ydata_synthetic/synthesizers/timeseries/timegan/model.py
"""
import os
import datetime as dt
from tqdm import tqdm, trange
from itertools import chain

import torch
from torch.nn import Module, Linear, GRU
import torch.nn.functional as F
from torch.optim import Optimizer, Adam
from torch.utils.data import Dataset, DataLoader
from timegan import RealDataset, StockData

INPUT_DIM = len(RealDataset.FEATURES)
HIDDEN_DIM = 24
NOISE_DIM = 32
BATCH_SIZE = 128
SEQUENCE_LENGTH = 30
TRAIN_STEPS = 0
G_LR = 5e-6
D_LR = 5e-6
GAMMA = 1

FOLDER = "gru2-stock"


class BasicGRU(Module):

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, dropout=0) -> None:
        super(BasicGRU, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout

        self.gru_layers = GRU(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, dropout=0 if num_layers == 1 else dropout, batch_first=True)
        self.output_layer = Linear(hidden_dim, output_dim)

    def forward(self, X):
        # X is (batch_size, timesteps, features)
        X, _ = self.gru_layers(X)
        if self.dropout > 0:
            torch.dropout(X, self.dropout, self.training)
        X = self.output_layer(X)
        return torch.sigmoid(X)

class Autoencoder(Module):

    def __init__(self, embedder: Module, recovery: Module) -> None:
        super(Autoencoder, self).__init__()
        self.embedder = embedder
        self.recovery = recovery

    def forward(self, X):
        H = self.embedder(X)
        X_tilde = self.recovery(H)
        return X_tilde

class Adversarial(Module):

    def __init__(self, generator_aux, discriminator, supervisor=None) -> None:
        super(Adversarial, self).__init__()
        self.generator_aux = generator_aux
        self.supervisor = supervisor
        self.discriminator = discriminator

    def forward(self, Z):
        E_Hat = self.generator_aux(Z)
        if self.supervisor is not None:
            H_hat = self.supervisor(E_Hat)
            Y_fake = self.discriminator(H_hat)
            return Y_fake
        Y_fake_e = self.discriminator(E_Hat)
        return Y_fake_e

class Generator(Module):

    def __init__(self, generator_aux, supervisor, recovery) -> None:
        super(Generator, self).__init__()
        self.generator_aux = generator_aux
        self.supervisor = supervisor
        self.recovery = recovery

    def forward(self, Z):
        E_Hat = self.generator_aux(Z)
        H_hat = self.supervisor(E_Hat)
        X_hat = self.recovery(H_hat)
        return X_hat

class DiscriminatorModel(Module):

    def __init__(self, embedder, discriminator) -> None:
        super(DiscriminatorModel, self).__init__()
        self.embedder = embedder
        self.discriminator = discriminator

    def forward(self, X):
        H = self.embedder(X)
        Y_real = self.discriminator(H)
        return Y_real

def train_autoencoder(x, optimizer: Optimizer, autoencoder: Autoencoder):
    optimizer.zero_grad()
    x_tilde = autoencoder(x)
    embedding_loss_t0 = F.mse_loss(x_tilde, x)
    e_loss_0 = 10 * torch.sqrt(embedding_loss_t0)

    e_loss_0.backward()
    optimizer.step()
    return torch.sqrt(embedding_loss_t0)

def train_supervisor(x, optimizer: Optimizer, embedder: Module, supervisor: Module):
    optimizer.zero_grad()
    h = embedder(x)
    h_hat_supervised = supervisor(h)
    generator_loss_supervised = F.mse_loss(h_hat_supervised[:, :-1, :], h[:, 1:, :])

    generator_loss_supervised.backward()
    optimizer.step()
    return generator_loss_supervised

def train_embedder(x, optimizer: Optimizer, embedder: Module, supervisor: Module, autoencoder: Autoencoder):
    optimizer.zero_grad()
    h = embedder(x)
    h_hat_supervised = supervisor(h)
    generator_loss_supervised = F.mse_loss(h_hat_supervised[:, :-1, :], h[:, 1:, :])

    x_tilde = autoencoder(x)
    embedding_loss_t0 = F.mse_loss(x_tilde, x)
    e_loss = 10 * torch.sqrt(embedding_loss_t0) + 0.1 * generator_loss_supervised

    e_loss.backward()
    optimizer.step()
    return torch.sqrt(embedding_loss_t0)

def discriminator_loss(x, z, discriminator_model: DiscriminatorModel, adversarial_supervised: Adversarial, adversarial_embedded: Adversarial, gamma: float):
    y_real = discriminator_model(x)
    discriminator_loss_real = F.binary_cross_entropy(y_real, torch.ones_like(y_real))

    y_fake = adversarial_supervised(z)
    discriminator_loss_fake = F.binary_cross_entropy(y_fake, torch.zeros_like(y_fake))

    y_fake_e = adversarial_embedded(z)
    discriminator_loss_fake_e = F.binary_cross_entropy(y_fake_e, torch.zeros_like(y_fake_e))

    return discriminator_loss_real + discriminator_loss_fake + gamma * discriminator_loss_fake_e

def calc_generator_moments_loss(y_true, y_pred):
    y_true_std, y_true_mean = torch.std_mean(y_true, dim=0)
    y_true_var = torch.square(y_true_std)
    y_pred_std, y_pred_mean = torch.std_mean(y_pred, dim=0)
    y_pred_var = torch.square(y_pred_std)
    g_loss_mean = torch.mean(torch.abs(y_true_mean - y_pred_mean))
    g_loss_var = torch.mean(torch.abs(torch.sqrt(y_true_var + 1e-6) - torch.sqrt(y_pred_var + 1e-6)))
    return g_loss_mean + g_loss_var

def train_generator(x, z, optimizer: Optimizer, adversarial_supervised: Adversarial, adversarial_embedded: Adversarial, embedder: Module, supervisor: Module, generator: Generator):
    optimizer.zero_grad()
    y_fake = adversarial_supervised(z)
    generator_loss_unsupervised = F.binary_cross_entropy(y_fake, torch.ones_like(y_fake))

    y_fake_e = adversarial_embedded(z)
    generator_loss_unsupervised_e = F.binary_cross_entropy(y_fake_e, torch.ones_like(y_fake_e))

    h = embedder(x)
    h_hat_supervised = supervisor(h)
    generator_loss_supervised = F.mse_loss(h_hat_supervised[:, :-1, :], h[:, 1:, :])

    x_hat = generator(z)
    generator_moment_loss = calc_generator_moments_loss(x, x_hat)

    generator_loss = generator_loss_unsupervised + generator_loss_unsupervised_e + 100 * torch.sqrt(generator_loss_supervised) + 100 * generator_moment_loss

    generator_loss.backward()
    optimizer.step()
    return generator_loss_unsupervised, generator_loss_supervised, generator_moment_loss

def train_discriminator(x, z, optimizer: Optimizer, discriminator_model: DiscriminatorModel, adversarial_supervised: Adversarial, adversarial_embedded: Adversarial, gamma: float):
    optimizer.zero_grad()
    disc_loss = discriminator_loss(x, z, discriminator_model, adversarial_supervised, adversarial_embedded, gamma)

    disc_loss.backward()
    optimizer.step()
    return disc_loss

def get_batch_data(ds: Dataset):
    return iter(DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True))

def get_batch_noise():
    return torch.rand((BATCH_SIZE, SEQUENCE_LENGTH, NOISE_DIM))

def sample(n_samples, generator: Generator):
    steps = n_samples // BATCH_SIZE + 1
    data = []
    for _ in trange(steps, desc="Synthetic data generation"):
        Z_ = get_batch_noise()
        records = generator(Z_).detach()
        data.append(records)
    return torch.vstack(data)

if __name__ == "__main__":
    generator_aux = BasicGRU(NOISE_DIM, HIDDEN_DIM, HIDDEN_DIM)
    supervisor = BasicGRU(HIDDEN_DIM, HIDDEN_DIM, HIDDEN_DIM, num_layers=2)
    discriminator = BasicGRU(HIDDEN_DIM, HIDDEN_DIM, output_dim=1)
    recovery = BasicGRU(HIDDEN_DIM, HIDDEN_DIM, INPUT_DIM)
    embedder = BasicGRU(INPUT_DIM, HIDDEN_DIM, HIDDEN_DIM)

    autoencoder = Autoencoder(embedder, recovery)
    adversarial_supervised = Adversarial(generator_aux, discriminator, supervisor=supervisor)
    adversarial_embedded = Adversarial(generator_aux, discriminator, supervisor=None)
    generator = Generator(generator_aux, supervisor, recovery)
    discriminator_model = DiscriminatorModel(embedder, discriminator)

    train_ds = RealDataset(os.path.join("data", "features.csv"), dt.datetime(1995, 3, 1), dt.datetime(2019, 12, 31), timesteps=SEQUENCE_LENGTH)
    # train_ds = StockData(os.path.join("data", "stock_data.csv"), timesteps=SEQUENCE_LENGTH)

    generator_aux.train()
    supervisor.train()
    discriminator.train()
    recovery.train()
    embedder.train()

    ## Embedding network training
    autoencoder_opt = Adam(chain(embedder.parameters(), recovery.parameters()), lr=G_LR)
    for _ in tqdm(range(TRAIN_STEPS), desc="Embedding network training"):
        X_ = next(get_batch_data(train_ds))
        step_e_loss_t0 = train_autoencoder(X_, autoencoder_opt, autoencoder)

    ## Supervised network training
    supervisor_opt = Adam(chain(supervisor.parameters(), generator.parameters()), lr=G_LR)
    for _ in tqdm(range(TRAIN_STEPS), desc="Supervised network training"):
        X_ = next(get_batch_data(train_ds))
        step_g_loss_s = train_supervisor(X_, supervisor_opt, embedder, supervisor)

    ## Joint training
    generator_opt = Adam(chain(generator_aux.parameters(), supervisor.parameters()), lr=G_LR)
    embedder_opt = Adam(chain(embedder.parameters(), recovery.parameters()), lr=G_LR)
    discriminator_opt = Adam(discriminator.parameters(), lr=D_LR)

    step_g_loss_u = step_g_loss_s = step_g_loss_v = step_e_loss_t0 = step_d_loss = 0
    for _ in tqdm(range(TRAIN_STEPS), desc="Joint networks training"):
        for _ in range(2):
            X_ = next(get_batch_data(train_ds))
            Z_ = get_batch_noise()

            step_g_loss_u, step_g_loss_S, step_g_loss_v = train_generator(X_,Z_, generator_opt, adversarial_supervised, adversarial_embedded, embedder, supervisor, generator)
            step_e_loss_t0 = train_embedder(X_, embedder_opt, embedder, supervisor, autoencoder)

        X_ = next(get_batch_data(train_ds))
        Z_ = get_batch_noise()
        step_d_loss = discriminator_loss(X_, Z_, discriminator_model, adversarial_supervised, adversarial_embedded, GAMMA)
        if step_d_loss > 0.15:
            step_d_loss = train_discriminator(X_, Z_, discriminator_opt, discriminator_model, adversarial_supervised, adversarial_embedded, GAMMA)

    if not os.path.exists(FOLDER):
        os.mkdir(FOLDER)
    torch.save(embedder.cpu().state_dict(), FOLDER + '/embedder.pt')
    torch.save(recovery.cpu().state_dict(), FOLDER +'/recovery.pt')
    torch.save(supervisor.cpu().state_dict(), FOLDER +'/supervisor.pt')
    torch.save(generator_aux.cpu().state_dict(), FOLDER + '/generator_aux.pt')
    torch.save(discriminator.cpu().state_dict(), FOLDER + '/discriminator.pt')
