from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import numpy as np
from timegan import *
import matplotlib.gridspec as gridspec

n_components = 2
seq_len = 30
sample_size = 100
file_path = 'gru'


if __name__ == "__main__":
    print("hello")
    
    X = RealDataset(os.path.join("data", "features.csv"), dt.datetime(1995, 1, 3), dt.datetime(2019, 12, 31))
    loader = DataLoader(X, batch_size = len(X))
    real_data = next(iter(loader)).numpy()
    real_data_reduced = real_data.reshape(-1, seq_len)
    
    
    
    encoder = BasicGRU(7, 12, 3)
    decoder = FFNN(12, 16, 7, 3)
    supervisor = BasicGRU(12, 12, 3)
    generator = BasicGRU(7, 12, 3)
    discriminator = GRUDiscriminator(12, 16, 3)
    
    encoder.load_state_dict(torch.load(file_path + '/encoder'))
    decoder.load_state_dict(torch.load(file_path + '/decoder'))
    supervisor.load_state_dict(torch.load(file_path + '/supervisor'))
    generator.load_state_dict(torch.load(file_path + '/generator'))
    discriminator.load_state_dict(torch.load(file_path + '/discriminator'))
    
    model = TimeGAN(encoder, decoder, generator, discriminator, supervisor)
    
    synthetic_sample = generate_data(real_data.shape[0], 30, model).numpy()
    synth_data_reduced = synthetic_sample.reshape(-1,seq_len)
    
    
    ######## pca #########     
    pca = PCA(n_components=n_components)
    pca.fit(real_data_reduced)

    pca_real = pd.DataFrame(pca.transform(real_data_reduced))
    pca_synth = pd.DataFrame(pca.transform(synth_data_reduced))

    fig = plt.figure(constrained_layout=True, figsize=(20, 10))
    spec = gridspec.GridSpec(ncols=2, nrows=1, figure=fig)

    ax = fig.add_subplot(spec[0,0])
    ax.set_title('PCA results',
             fontsize=20,
             color='red',
             pad=10)

    # PCA scatter plot
    plt.scatter(pca_real.iloc[:, 0].values, pca_real.iloc[:, 1].values,
            c='black', alpha=0.2, label='Original')

    plt.scatter(pca_synth.iloc[:, 0], pca_synth.iloc[:, 1],
            c='red', alpha=0.2, label='Synthetic')
    
    ax.legend()
    plt.show()
    
    ########## tnse ###########
    data_reduced = np.concatenate((real_data_reduced, synth_data_reduced), axis=0)
    tsne = TSNE(n_components=n_components, n_iter=300)
    tsne_results = pd.DataFrame(tsne.fit_transform(data_reduced))
    
    fig = plt.figure(constrained_layout=True, figsize=(20, 10))
    spec = gridspec.GridSpec(ncols=2, nrows=1, figure=fig)

    ax2 = fig.add_subplot(spec[0,0])
    ax2.set_title('TSNE results',
              fontsize=20,
              color='red',
              pad=10)

    # t-SNE scatter plot
    plt.scatter(tsne_results.iloc[:700, 0].values, tsne_results.iloc[:700, 1].values,
            c='black', alpha=0.2, label='Original')
    plt.scatter(tsne_results.iloc[700:, 0], tsne_results.iloc[700:, 1],
            c='red', alpha=0.2, label='Synthetic')

    ax2.legend()
    plt.show()
    
    
    
    