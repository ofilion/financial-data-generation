
import os
import datetime as dt
from typing import Callable
from itertools import chain

import torch
from torch.nn import Module, GRU, Linear, MSELoss, ModuleList, BCEWithLogitsLoss
from torch.optim import Optimizer, Adam
from torch.utils.data import Dataset, DataLoader
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np

EPOCHS = 50
BATCH_SIZE = 128
G_LR = 1e-3
D_LR = 1e-5
HIDDEN_SIZE = 12
HIDDEN_SIZE_2 = 16
GRU_LAYERS = 3
DISC_LAYERS = 3
FF_LAYERS = 3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### transformer imports
from transformer import *

class RealDataset(Dataset):

    FEATURES = ["Return", "Open-Close", "Open-Low", "Open-High", "Normalized Volume", "VIX", "VIX Open-Close"]

    def __init__(self, filepath: str, start_date, end_date, timesteps=30) -> None:
        super(RealDataset, self).__init__()
        self.df = pd.read_csv(filepath, parse_dates=True, index_col="Date").loc[start_date:end_date]
        self.timesteps = timesteps

        self.df = self.df[RealDataset.FEATURES]
        self.mean = self.df.mean()
        self.std = self.df.std(ddof=1)
        self.norm_df = (self.df - self.mean) / self.std

    def __len__(self):
        return len(self.df) - self.timesteps

    def __getitem__(self, index) -> torch.Tensor:
        return torch.from_numpy(self.norm_df.iloc[index:index+self.timesteps].values).float()

class TimeGAN(Module):

    def __init__(self, encoder, decoder, generator, discriminator, supervisor) -> None:
        super(TimeGAN, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.generator = generator
        self.discriminator = discriminator
        self.supervisor = supervisor

    def forward(self, noise):
        return self.decoder(self.generator(noise))
    
class BasicGRU(Module):

    def __init__(self, input_size, hidden_size, num_layers, bidirectional=False) -> None:
        super().__init__()
        self.rnn = GRU(input_size, hidden_size, num_layers, batch_first=True,
                       bidirectional=bidirectional)
        self.linear = Linear(hidden_size * (2 if bidirectional else 1), hidden_size)

    def forward(self, X):
        X, _ = self.rnn(X)
        return torch.sigmoid(self.linear(X))

class GRUDiscriminator(Module):

    def __init__(self, input_size, hidden_size, num_layers) -> None:
        super(GRUDiscriminator, self).__init__()
        self.gru = BasicGRU(input_size, hidden_size, num_layers, bidirectional=True)
        self.lin1 = Linear(hidden_size, hidden_size)
        self.out = Linear(hidden_size, 1)

    def forward(self, X):
        X = self.gru(X)
        X = torch.mean(X, dim=1)    # Average accross timesteps

        X = self.lin1(X)
        X = torch.relu(X)
        return self.out(X)

class FFNN(Module):

    def __init__(self, input_size, hidden_size, output_size, num_layers) -> None:
        super(FFNN, self).__init__()
        self.layers = ModuleList([Linear(input_size if i == 0 else hidden_size,
                                         output_size if i == num_layers - 1 else hidden_size)
                                  for i in range(num_layers)])

    def forward(self, X):
        for layer in self.layers[:-1]: 
            X = layer(X)
           
            F.leaky_relu(X)
        return self.layers[-1](X)

def train_autoencoder_step(X: torch.Tensor, model: TimeGAN, criterion: Callable,
                           optimizer: Optimizer) -> float:
    optimizer.zero_grad()
    # X is (batch_size, timesteps, features)
    n_features = X.shape[-1]
    encoded = model.encoder(X)
    decoded = model.decoder(encoded)

    loss = criterion(decoded.view(-1, n_features), X.view(-1, n_features))
    loss.backward()
    optimizer.step()
    return loss.detach().item()

def train_superviser_step(X: torch.Tensor, model: TimeGAN, criterion: Callable,
                          optimizer: Optimizer) -> float:
    optimizer.zero_grad()
    encoded = model.encoder(X)

    y = encoded[:, 1:, :]

    y_hat = model.supervisor(encoded)[:,:-1,:]
   # print(y.shape, y_hat.shape )
    loss = criterion(torch.reshape(y_hat, (-1, y.shape[-1])), torch.reshape(y , (-1, y.shape[-1])))
    loss.backward()
    optimizer.step()
    return loss.detach().item()

def train_joint_autoencoder_supervisor_step(X: torch.Tensor, model: TimeGAN, sup_criterion: Callable,
                                            recons_criterion: Callable, optimizer: Optimizer) -> float:
    optimizer.zero_grad()
    # X is (batch_size, timesteps, features)
    n_features = X.shape[-1]

    encoded = model.encoder(X)

    # 1. Supervisor
    y = encoded[:, 1:, :]
    y_hat = model.supervisor(encoded)[:, :-1, :]
    sup_loss = sup_criterion(torch.reshape(y_hat, (-1, y.shape[-1])),torch.reshape(y, (-1, y.shape[-1])))

    # 2. Autoencoder
    decoded = model.decoder(encoded)
    recons_loss = recons_criterion(torch.reshape(decoded, (-1, n_features)), torch.reshape(X, (-1, n_features)))

    total_loss = 10 * torch.sqrt(recons_loss) + 0.1 * sup_loss
    total_loss.backward()
    optimizer.step()
    return total_loss.detach().item()

def train_joint_generator_supervisor_step(X: torch.Tensor, Z: torch.Tensor, model: TimeGAN,
                                          sup_criterion: Callable,  disc_criterion: Callable,
                                          optimizer: Optimizer) -> float:
    optimizer.zero_grad()
    generated_latent_space = model.generator(Z)
    supervised_latent_space = model.supervisor(generated_latent_space)
    y_hat_supervised = model.discriminator(supervised_latent_space)
    loss1 = disc_criterion(y_hat_supervised, torch.ones_like(y_hat_supervised))

    y_hat = model.discriminator(generated_latent_space)
    loss2 = disc_criterion(y_hat, torch.ones_like(y_hat))
    disc_loss = loss1 + loss2

    encoded_X = model.encoder(X)
    supervisor_y = encoded_X[:, 1:, :]
    supervisor_y_hat = model.supervisor(encoded_X)[:, :-1, :]
    supervisor_loss = sup_criterion(torch.reshape(supervisor_y_hat, (-1, supervisor_y.shape[-1])),
                                    torch.reshape(supervisor_y, (-1, supervisor_y.shape[-1])))
    
    x_hat = model.decoder(supervised_latent_space)
    # x_hat is (batch_size, timesteps, features)
    real_std, real_mean = torch.std_mean(X.view(-1, X.shape[-1]), dim=0)
    pred_std, pred_mean = torch.std_mean(x_hat.view(-1, x_hat.shape[-1]), dim=0)
    mean_loss = torch.mean(torch.abs(real_mean - pred_mean))
    std_loss = torch.mean(torch.abs(real_std - pred_std))
    moments_loss = mean_loss + std_loss

    total_loss = disc_loss + 100 * torch.sqrt(supervisor_loss) + 100 * moments_loss
    total_loss.backward()
    optimizer.step()
    return total_loss.detach().item()

def train_discriminator_step(X: torch.Tensor, Z: torch.Tensor, model: TimeGAN, criterion: Callable,
                             optimizer: Optimizer) -> float:
    optimizer.zero_grad()
    encoded_real = model.encoder(X)
    y_real = model.discriminator(encoded_real)
    real_loss = criterion(y_real, torch.ones_like(y_real))

    encoded_noise = model.generator(Z)
    supervised_noise = model.supervisor(encoded_noise)
    y_fake_sup = model.discriminator(supervised_noise)
    y_fake = model.discriminator(encoded_noise)

    fake_loss_sup = criterion(y_fake_sup, torch.zeros_like(y_fake_sup))
    fake_loss = criterion(y_fake, torch.zeros_like(y_fake))

    total_loss = real_loss + fake_loss_sup + fake_loss
    total_loss.backward()
    optimizer.step()
    return total_loss.detach().item()

def train(model: TimeGAN, X_ds: Dataset, epochs: int, lr: float, d_lr: float, batch_size: int):

    X_loader = DataLoader(X_ds, batch_size, shuffle=True)
    mse_loss = MSELoss()
    bce_loss = BCEWithLogitsLoss()
    
    print("\nTraining autoencoder")
    autoencoder_optimizer = Adam(chain(model.encoder.parameters(), model.decoder.parameters()), lr)
    for e in range(epochs//2):
        running_loss = 0.
        for X in X_loader:
            X = X.to(DEVICE)
            running_loss += train_autoencoder_step(X, model, mse_loss, autoencoder_optimizer)
        print(f"Epoch {e+1}: Loss = {running_loss/len(X_loader)}")

    print("\nTraining supervisor")
    supervisor_optimizer = Adam(chain(model.encoder.parameters(), model.supervisor.parameters()), lr)
    for e in range(epochs//2):
        running_loss = 0.
        for X in X_loader:
            X = X.to(DEVICE)
            running_loss += train_superviser_step(X, model, mse_loss, supervisor_optimizer)
        print(f"Epoch {e+1}: Loss = {running_loss/len(X_loader)}")

    print("\nJoint training")
    as_optimizer = Adam(chain(model.encoder.parameters(), model.decoder.parameters()), lr)
    gs_optimizer = Adam(chain(model.generator.parameters(), model.supervisor.parameters()), lr)
    disc_optimizer = Adam(model.discriminator.parameters(), d_lr)
    for e in range(epochs):
        running_gs_loss = 0.
        running_as_loss = 0.
        running_disc_loss = 0.
        for X in X_loader:
            X = X.to(DEVICE)
            Z = torch.randn_like(X)
            gs_loss = 0.
            as_loss = 0.
            k = 3
            for _ in range(k):
                gs_loss += train_joint_generator_supervisor_step(X, Z, model, mse_loss, mse_loss, gs_optimizer)
                as_loss += train_joint_autoencoder_supervisor_step(X, model, mse_loss, mse_loss, as_optimizer)
            running_gs_loss += gs_loss / k
            running_as_loss += as_loss / k

            running_disc_loss += train_discriminator_step(X, Z, model, bce_loss, disc_optimizer)

        print(f"Epoch {e+1}: AS Loss = {running_as_loss/len(X_loader)}, GS Loss = {running_gs_loss/len(X_loader)}, Discriminator Loss = {running_disc_loss/len(X_loader)}")

def generate_data(n_points: int, timesteps: int, model: TimeGAN) -> torch.Tensor:
    Z = torch.randn((n_points, timesteps, len(RealDataset.FEATURES))).to(DEVICE)
    latent = model.generator(Z)
    return model.decoder(latent).detach()


def visualize(generated_data, real_data, cols):

    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 10))
    axes=axes.flatten()
   
    time = list(range(1,30))
    for j, col in enumerate(cols):
    
        frame = pd.DataFrame({'Real': real_data[:, j],
                   'Synthetic': generated_data[:, j]})
        frame.plot(ax=axes[j],
                   title = col,
                   secondary_y='Synthetic data', style=['-', '--'])
    fig.tight_layout()




if __name__ == "__main__":
    print("Using ", DEVICE)
    X = RealDataset(os.path.join("data", "features.csv"), dt.datetime(1995, 1, 3), dt.datetime(2019, 12, 31))

    print(X[0].shape)
  
    encoder = BasicGRU(len(RealDataset.FEATURES), HIDDEN_SIZE, GRU_LAYERS)
    decoder = FFNN(HIDDEN_SIZE, HIDDEN_SIZE_2, len(RealDataset.FEATURES), FF_LAYERS)
    supervisor = BasicGRU(HIDDEN_SIZE, HIDDEN_SIZE, GRU_LAYERS)
    generator = BasicGRU(len(RealDataset.FEATURES), HIDDEN_SIZE, GRU_LAYERS)
    discriminator = GRUDiscriminator(HIDDEN_SIZE, HIDDEN_SIZE_2, GRU_LAYERS)
  
    '''
    #num_hidden, hidden_size, intermediate_size, num_heads, dropout_prob, seq_len
    encoder = TransformerEncoder(4,9,6,3,30)
    #input_size, hidden_size, output_size, num_layers
    decoder = FFNN(9, 14, 7, 3)
    supervisor = TransformerEncoder(4,9,6,3,30,False)
   # supervisor = TransformerForPrediction(supervisor_encoder)
    generator = TransformerEncoder(4,9,6,3,30)
    
    discriminator_encoder = TransformerEncoder(2,9,4,3,30,False)
    discriminator = TransformerForBinaryClassification(discriminator_encoder)
    '''

    
    X_loader = DataLoader(X, 1, shuffle=True)
    example = next(iter(X_loader))
    print("here",example[0].shape)


    model = TimeGAN(encoder, decoder, generator, discriminator, supervisor)
    train(model, X, 50, 1e-3, 5e-4, 128)
    
    
    folder_name = 'gru'
    if not os.path.exist(folder_name):
        os.makedir(folder_name)
    torch.save(model.encoder.state_dict(), folder_name + '/encoder.pt')
    torch.save(model.decoder.state_dict(), folder_name +'/decoder.pt')
    torch.save(model.supervisor.state_dict(), folder_name +'/supervisor.pt')
    torch.save(model.generator.state_dict(), folder_name + '/generator.pt')
    torch.save(model.discriminator.state_dict(), folder_name + '/discriminator.pt')
    

    generated_data = generate_data(3, 30, model)
    generated_data * torch.from_numpy(X.std.values) + torch.from_numpy(X.mean.values)
    print(generated_data[0].shape)
    
    cols = [
    "Return","Open-Close",'Open-Low',"Open-High","Normalized Volume", "VIX", "VIX Open-close"
]
    visualize(generated_data[0].detach(), example[0].detach(), cols)
    
