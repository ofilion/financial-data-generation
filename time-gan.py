
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

"pip install ydata-synthetic==0.3.0"
from ydata_synthetic.synthesizers.timeseries import TimeGAN

def prep_data(data: np.array, window_len, scaler) -> np.array:
    
    """
    Args:
      -  data = np.array
      -  window_len = length of window
      -  scaler = sklearn.preprocessing

    Returns:
      - processed: preprocessed data as python list
    """
    # normalize data
    scaler = scaler.fit(data)
    scaled_data = scaler.transform(data)
    
    # group data into windows of length window_len
    windows = []
    for i in range(len(data) - window_len):
        windows.append(scaled_data[i:i+window_len])
        
    # reorder the data
    idx = np.random.permutation(len(windows))

    processed = []
    for i in range(len(windows)):
        processed.append(windows[idx[i]])
    
    return processed

if __name__ == "__main__":
    df = pd.read_csv("./data/features.csv")
    df['Date'] = pd.to_datetime(df['Date'])

    seq_len = 30        # Timesteps
    n_seq = 11          # Features

    hidden_dim = 24     # Hidden units for generator (GRU & LSTM)

    gamma = 1           # discriminator loss

    noise_dim = 32      # Used by generator as a starter dimension
    dim = 128           # UNUSED
    batch_size = 128

    learning_rate = 5e-4
    beta_1 = 0          # UNUSED
    beta_2 = 1          # UNUSED
    data_dim = 28       # UNUSED

    # batch_size, lr, beta_1, beta_2, noise_dim, data_dim, layers_dim
    gan_args = [batch_size, learning_rate, beta_1, beta_2, noise_dim, data_dim, dim]

    # define minmax scaler
    scaler = MinMaxScaler()

    # set index to date
    try:
        df = df.set_index('Date').sort_index()
    except:
        df = df

    # prep data
    data = prep_data(df.values, 30, scaler)

    synth = TimeGAN(model_parameters=gan_args, hidden_dim=hidden_dim, seq_len=seq_len, n_seq=n_seq, gamma=1)
    synth.train(energy_data, train_steps=500)
    synth.save('synth.pkl')

    synth_data = synth.sample(len(energy_data))


