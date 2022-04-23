from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import numpy as np
from timegan import *
import matplotlib as plt
import matplotlib.gridspec as gridspec

n_components = 2
seq_len = 30
sample_size = 100
file_path = 'gru'

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    print("hello")
    
    X = RealDataset(os.path.join("data", "features.csv"), dt.datetime(1995, 1, 3), dt.datetime(2019, 12, 31))
    loader = DataLoader(X, batch_size = len(X))
    real_data = next(iter(loader)).cpu().numpy()[0:sample_size]
    real_data_reduced = real_data.reshape(-1, seq_len)
    
    input_size = len(RealDataset.FEATURES)
    encoder = BasicGRU(input_size, HIDDEN_SIZE, HIDDEN_SIZE, GRU_LAYERS)
    decoder = FFNN(HIDDEN_SIZE, HIDDEN_SIZE, input_size, FF_LAYERS)
    supervisor = BasicGRU(HIDDEN_SIZE, HIDDEN_SIZE, HIDDEN_SIZE, GRU_LAYERS)
    generator = BasicGRU(input_size, HIDDEN_SIZE, HIDDEN_SIZE, GRU_LAYERS)
    discriminator = GRUDiscriminator(HIDDEN_SIZE, HIDDEN_SIZE, GRU_LAYERS)
    
    encoder.load_state_dict(torch.load(file_path + '/encoder.pt'))
    decoder.load_state_dict(torch.load(file_path + '/decoder.pt'))
    supervisor.load_state_dict(torch.load(file_path + '/supervisor.pt'))
    generator.load_state_dict(torch.load(file_path + '/generator.pt'))
    discriminator.load_state_dict(torch.load(file_path + '/discriminator.pt'))
    
    model = TimeGAN(encoder, decoder, generator, discriminator, supervisor).to(DEVICE)
    
    synthetic_sample = generate_data(sample_size, 30, model).cpu().numpy()
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
    #plt.savefig("pca.png")
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
    #plt.savefig("tsne.png")
    plt.show()
    
    
    
    