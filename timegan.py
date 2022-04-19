
import os
import datetime as dt
from typing import Callable
import torch
from torch.nn import Module, GRU, Linear, MSELoss, ModuleList
import torch.nn.functional as F
from torch.optim import Optimizer, Adam
from torch.utils.data import Dataset, DataLoader
import pandas as pd


### transformer imports
from transformer import *

class RealDataset(Dataset):

    def __init__(self, filepath: str, start_date, end_date, timesteps=30) -> None:
        super(RealDataset, self).__init__()
        self.df = pd.read_csv(filepath, parse_dates=True, index_col="Date").loc[start_date:end_date]
        self.timesteps = timesteps

    def __len__(self):
        return len(self.df) - self.timesteps

    def __getitem__(self, index) -> torch.Tensor:
        return torch.from_numpy(self.df[["Return", "Open-Close", "Open-Low", "Open-High", "Normalized Volume", "VIX", "VIX Open-Close"]].iloc[index:index+self.timesteps].values).float()

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
        self.linear = Linear(hidden_size, hidden_size)

    def forward(self, X):
        X, _ = self.rnn(X)
        return F.sigmoid(self.linear(X))

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
    supervisor_input = encoded[:, :-1, :]

    y_hat = model.supervisor(supervisor_input)

    loss = criterion(y_hat.view(-1, y.shape[-1]), y.view(-1, y.shape[-1]))
    loss.backward()
    optimizer.step()
    return loss.detach.item()

def train_joint_autoencoder_supervisor_step(X: torch.Tensor, model: TimeGAN, sup_criterion: Callable,
                                            recons_criterion: Callable, optimizer: Optimizer) -> float:
    optimizer.zero_grad()
    # X is (batch_size, timesteps, features)
    n_features = X.shape[-1]

    encoded = model.encoder(X)

    # 1. Supervisor
    supervisor_input = encoded[:, :-1, :]
    y = encoded[:, 1:, :]
    y_hat = model.supervisor(supervisor_input)
    sup_loss = sup_criterion(y_hat.view(-1, y.shape[-1]), y.view(-1, y.shape[-1]))

    # 2. Autoencoder
    decoded = model.decoder(encoded)
    recons_loss = recons_criterion(decoded.view(-1, n_features), X.view(-1, n_features))

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
    supervisor_input = encoded_X[:, :-1, :]
    supervisor_y = encoded_X[:, 1:, :]
    supervisor_y_hat = model.supervisor(supervisor_input)
    supervisor_loss = sup_criterion(supervisor_y_hat.view(-1, supervisor_y.shape[-1]),
                                    supervisor_y.view(-1, supervisor_y.shape[-1]))
    
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

def train(model: TimeGAN, X_ds: Dataset, epochs: int, lr: float, batch_size: int):

    X_loader = DataLoader(X_ds, batch_size, shuffle=True)
    
    print("Training autoencoder")
    autoencoder_loss = MSELoss()
    autoencoder_optimizer = Adam(list(model.encoder.parameters()) + list(model.decoder.parameters()), lr)
    for e in range(epochs):
        running_loss = 0.
        for X in X_loader:
            running_loss += train_autoencoder_step(X, model, autoencoder_loss, autoencoder_optimizer)
        print(f"Epoch {e+1}: Loss = {running_loss/len(X_loader)}")


if __name__ == "__main__":
    X = RealDataset(os.path.join("data", "features.csv"), dt.datetime(1995, 1, 3), dt.datetime(2019, 12, 31))

  
  #  encoder = BasicGRU(7, 4, 1)
  #  decoder = FFNN(4, 8, 7, 3)
 
  
 #num_hidden, hidden_size, intermediate_size, num_heads, dropout_prob, seq_len
    encoder = TransformerEncoder(3,9,4,3,0.3,30)
    #input_size, hidden_size, output_size, num_layers
    decoder = FFNN(9, 6, 7, 3)
  
    
    model = TimeGAN(encoder, decoder, None, None, None)
    train(model, X, 10, 0.01, 128)
