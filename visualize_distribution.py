from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import numpy as np
# from timegan import *
from timegan2 import *
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import pandas as pd

n_components = 2
seq_len = 30
sample_size = 100
file_path = 'gru2-stock'


if __name__ == "__main__":
   
    
    X = RealDataset(os.path.join("data", "features.csv"), dt.datetime(1995, 1, 3), dt.datetime(2019, 12, 31))
    loader = DataLoader(X, batch_size = len(X))
    real_data = next(iter(loader)).numpy()[:sample_size]
    real_data_reduced = real_data.reshape(-1, SEQUENCE_LENGTH * INPUT_DIM)
    input_size = len(RealDataset.FEATURES)
    

    
    
#     encoder = BasicGRU(input_size, HIDDEN_SIZE, HIDDEN_SIZE, GRU_LAYERS)
#     # decoder = FFNN(HIDDEN_SIZE, HIDDEN_SIZE, input_size, FF_LAYERS)
#     decoder = BasicGRU(HIDDEN_SIZE, HIDDEN_SIZE, input_size, GRU_LAYERS)
#     supervisor = BasicGRU(HIDDEN_SIZE, HIDDEN_SIZE, HIDDEN_SIZE, GRU_LAYERS)
#     generator = BasicGRU(input_size, HIDDEN_SIZE, HIDDEN_SIZE, GRU_LAYERS)
#     # discriminator = GRUDiscriminator(HIDDEN_SIZE, HIDDEN_SIZE, GRU_LAYERS)
#     discriminator = BasicGRU(HIDDEN_SIZE, HIDDEN_SIZE, 1, DISC_LAYERS, bidirectional=True)
    
    
    '''
    encoder = TransformerEncoder(4,7,30,3,30)
    decoder = FFNN(7, 30, 7, 3)
    supervisor = TransformerEncoder(4,7,30,3,30)
    generator = TransformerEncoder(4,7,30,3,30)
    discriminator_encoder = TransformerEncoder(2,7,30,3,30)
    discriminator = TransformerForBinaryClassification(discriminator_encoder)
    '''
    
#     encoder.load_state_dict(torch.load(file_path + '/encoder.pt'))
#     decoder.load_state_dict(torch.load(file_path + '/decoder.pt'))
#     supervisor.load_state_dict(torch.load(file_path + '/supervisor.pt'))
#     generator.load_state_dict(torch.load(file_path + '/generator.pt'))
#     discriminator.load_state_dict(torch.load(file_path + '/discriminator.pt'))
    
#     model = TimeGAN(encoder, decoder, generator, discriminator, supervisor)
    
#     synthetic_sample = generate_data(sample_size, 30, model).numpy()

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

    generator_aux.load_state_dict(torch.load(os.path.join(file_path, "generator_aux.pt")))
    supervisor.load_state_dict(torch.load(os.path.join(file_path, "supervisor.pt")))
    discriminator.load_state_dict(torch.load(os.path.join(file_path, "discriminator.pt")))
    recovery.load_state_dict(torch.load(os.path.join(file_path, "recovery.pt")))
    embedder.load_state_dict(torch.load(os.path.join(file_path, "embedder.pt")))

    synthetic_sample = sample(sample_size, generator).numpy()
    synth_data_reduced = synthetic_sample.reshape(-1, SEQUENCE_LENGTH * INPUT_DIM)
    
    
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
    plt.scatter(tsne_results.iloc[:sample_size, 0].values, tsne_results.iloc[:sample_size, 1].values,
            c='black', alpha=0.2, label='Original')
    plt.scatter(tsne_results.iloc[sample_size:, 0], tsne_results.iloc[sample_size:, 1],
            c='red', alpha=0.2, label='Synthetic')

    ax2.legend()
    plt.show()
    
    
    ######## visualize ########
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 10))
    axes=axes.flatten()
   
    cols = [
    "Return","Open-Close",'Open-Low',"Open-High","Normalized Volume", "VIX", "VIX Open-close"
]
  
   
    time = list(range(1,30))
    for j, col in enumerate(cols):
    
        frame = pd.DataFrame({'Real': real_data[0,:, j],
                   'Synthetic': synthetic_sample[0,:, j]})
        frame.plot(ax=axes[j],
                   title = col,
                   secondary_y='Synthetic data', style=['-', '--'])
    fig.tight_layout()
    plt.show()
    
    
    
    
    
    
    
    
