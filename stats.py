import pickle as pkl
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from timegan2 import *
import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns
import pandas as pd

if __name__ == "__main__":
    
    generated = []
    for i in range(5):
        file_path = 'transformer'+str(i+1)+'/generated.pkl'
        file = open(file_path,'rb')
        arr = pkl.load(file)
        file.close()
        generated.append(arr)
    
    all_generated = np.concatenate(generated)
    print(all_generated.shape)

    with open("generated.pkl", "rb") as file:
        all_generated_gru = np.array(pkl.load(file))
    print(all_generated_gru.shape)

    with open("gan2.pkl", "rb") as file:
        all_generated_basic = np.array(pkl.load(file))
    print(all_generated_basic.shape)

    X = RealDataset(os.path.join("data", "features.csv"), dt.datetime(1995, 1, 3), dt.datetime(2019, 12, 31))
    loader = DataLoader(X, batch_size = len(X))
    real_data = next(iter(loader)).numpy()
    print(real_data.shape)

    print()

    scale_max = X.max.values[None, None, :]
    scale_min = X.min.values[None, None, :]
    data_mean = X.df[RealDataset.FEATURES].mean().values[None, None, :]
    data_std = X.df[RealDataset.FEATURES].std(ddof=1).values[None, None, :]

    real_data = real_data * (scale_max - scale_min) + scale_min
    all_generated =  all_generated * (scale_max - scale_min) + scale_min
    all_generated_gru =  all_generated_gru * (scale_max - scale_min) + scale_min
    all_generated_basic = all_generated_basic * data_std + data_mean

    ### PCA ###
    pca = PCA(n_components=2)
    
    pca_real = pd.DataFrame(pca.fit_transform(np.reshape(real_data, (real_data.shape[0], -1)))).sample(200)
    pca_trans = pd.DataFrame(pca.transform(np.reshape(all_generated, (all_generated.shape[0], -1)))).sample(200)
    pca_gru = pd.DataFrame(pca.transform(np.reshape(all_generated_gru, (all_generated_gru.shape[0], -1)))).sample(200)
    pca_basic = pd.DataFrame(pca.transform(np.reshape(all_generated_basic, (all_generated_basic.shape[0], -1)))).sample(200)

    fig = plt.figure(constrained_layout=True, figsize=(10, 10))
    spec = gridspec.GridSpec(ncols=1, nrows=1, figure=fig)
    
    ax = fig.add_subplot(spec[0,0])
    #ax.set_title('PCA results', fontsize=20, pad=10)

    plt.scatter(pca_real.iloc[:, 0].values, pca_real.iloc[:, 1].values, c='black', alpha=0.2, label='Original')
    plt.scatter(pca_basic.iloc[:, 0].values, pca_basic.iloc[:, 1].values, c='blue', alpha=0.2, label='Basic TimeGAN')
    plt.scatter(pca_gru.iloc[:, 0].values, pca_gru.iloc[:, 1].values, c='red', alpha=0.2, label='GRU TimeGAN')
    plt.scatter(pca_trans.iloc[:, 0].values, pca_trans.iloc[:, 1].values, c='green', alpha=0.2, label='Transformer TimeGAN')

    ax.legend()
    plt.savefig("pca.png")
    plt.cla()

    ### t-SNE ###
    data = np.concatenate([
        np.reshape(real_data[np.random.choice(real_data.shape[0], size=200, replace=False)], (200, -1)),
        np.reshape(all_generated_basic[np.random.choice(all_generated_basic.shape[0], size=200, replace=False)], (200, -1)),
        np.reshape(all_generated_gru[np.random.choice(all_generated_gru.shape[0], size=200, replace=False)], (200, -1)),
        np.reshape(all_generated[np.random.choice(all_generated.shape[0], size=200, replace=False)], (200, -1))
    ], axis=0)
    tsne = TSNE(n_components=2, n_iter=300)
    tsne_results = pd.DataFrame(tsne.fit_transform(data))

    fig = plt.figure(constrained_layout=True, figsize=(10, 10))
    spec = gridspec.GridSpec(ncols=1, nrows=1, figure=fig)

    ax2 = fig.add_subplot(spec[0,0])
    #ax2.set_title('TSNE results', fontsize=20, pad=10)

    plt.scatter(tsne_results.iloc[:200, 0].values, tsne_results.iloc[:200, 1].values, c='black', alpha=0.2, label='Original')
    plt.scatter(tsne_results.iloc[200:400, 0].values, tsne_results.iloc[200:400, 1].values, c='blue', alpha=0.2, label='Basic GAN')
    plt.scatter(tsne_results.iloc[400:600, 0].values, tsne_results.iloc[400:600, 1].values, c='red', alpha=0.2, label='GRU TimeGAN')
    plt.scatter(tsne_results.iloc[600:, 0].values, tsne_results.iloc[600:, 1].values, c='green', alpha=0.2, label='Transformer TimeGAN')

    ax2.legend()
    plt.savefig("tsne.png")
    plt.cla()

    ### Visualize an example ###
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 10))
    axes=axes.flatten()
   
    cols = [
    "Return","Open-Close",'Open-Low',"Open-High","Normalized Volume", "VIX", "VIX Open-close"
    ]
  
   
    time = list(range(1,30))
    idx = 2
    for j, col in enumerate(cols):
    
        frame = pd.DataFrame({'Real': real_data[idx,:, j],
                              #'Basic': all_generated_basic[idx,:,j],
                              'GRU':  all_generated_gru[idx,:,j],
                              'Transformer': all_generated[idx,:,j],
                              })
        frame.plot(ax=axes[j],
                   title = col,
                   secondary_y='Synthetic data', style=['-', '--', '--'])
    fig.tight_layout()
    plt.show()

    #####
    
    real_data = np.reshape(real_data ,(-1,7))
    all_generated_basic = np.reshape(all_generated_basic, (-1, 7))
    all_generated = np.reshape(all_generated ,(-1,7))
    all_generated_gru = np.reshape(all_generated_gru, (-1, 7))

    ##### calc means ######
    avg_real = np.mean(real_data,0)
    avg_generated = np.mean(all_generated,0)
    avg_gru = np.mean(all_generated_gru,0)
    avg_basic = np.mean(all_generated_basic,0)
    
    print("Means")
    print(avg_real)
    print(avg_basic)
    print(avg_gru)
    print(avg_generated)
    print()
    ##### calc var #####
    var_real = np.std(real_data,0, ddof=1)
    var_generated = np.std(all_generated,0, ddof=1)
    var_gru = np.std(all_generated_gru,0,ddof=1)
    var_basic = np.std(all_generated_basic,0,ddof=1)
    
    print("Standard deviations")
    print(var_real)
    print(var_basic)
    print(var_gru)
    print(var_generated)
    
    #### calc cov ######
    cov_matrix_real = np.corrcoef(real_data, rowvar=False)
    cov_matrix_generated = np.corrcoef(all_generated, rowvar=False)
    cov_matrix_gru = np.corrcoef(all_generated_gru, rowvar=False)
    cov_matrix_basic = np.corrcoef(all_generated_basic, rowvar=False)
    
    
    cols = [
    "Return","Open-Close",'Open-Low',"Open-High","Norm. Vol.", "VIX", "VIX Open-close"
    ]
    
    f, ax = plt.subplots(figsize=(11, 15)) 
    heatmap_real = sns.heatmap(cov_matrix_real, 
                      square = True,
                      linewidths = .5,
                      cmap = 'coolwarm',
                      cbar_kws = {'shrink': .4, 
                                'ticks' : [-1, -.5, 0, 0.5, 1]},
                      vmin = -1, 
                      vmax = 1,
                      annot = True,
                      annot_kws = {'size': 12})
  
    
    ax.set_yticklabels(cols, rotation = 0)
    ax.set_xticklabels(cols, rotation=25)
    ax.set_ylim(len(cols), -1)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    sns.set_style({'xtick.bottom': True}, {'ytick.left': True})
    plt.savefig("corr_real.png")
    plt.cla()
    
    f, ax = plt.subplots(figsize=(11, 15)) 
    heatmap_generated = sns.heatmap(cov_matrix_generated, 
                      square = True,
                      linewidths = .5,
                      cmap = 'coolwarm',
                      cbar_kws = {'shrink': .4, 
                                'ticks' : [-1, -.5, 0, 0.5, 1]},
                      vmin = -1, 
                      vmax = 1,
                      annot = True,
                      annot_kws = {'size': 12})
  
    
    ax.set_yticklabels(cols, rotation = 0)
    ax.set_xticklabels(cols,rotation=25)
    ax.set_ylim(len(cols), -1)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    sns.set_style({'xtick.bottom': True}, {'ytick.left': True})
    plt.savefig("corr_fake.png")
    plt.cla()
    
    f, ax = plt.subplots(figsize=(11, 15)) 
    heatmap_generated = sns.heatmap(cov_matrix_gru, 
                      square = True,
                      linewidths = .5,
                      cmap = 'coolwarm',
                      cbar_kws = {'shrink': .4, 
                                'ticks' : [-1, -.5, 0, 0.5, 1]},
                      vmin = -1, 
                      vmax = 1,
                      annot = True,
                      annot_kws = {'size': 12})
  
    
    ax.set_yticklabels(cols, rotation = 0)
    ax.set_xticklabels(cols,rotation=25)
    ax.set_ylim(len(cols), -1)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    sns.set_style({'xtick.bottom': True}, {'ytick.left': True})
    plt.savefig("corr_gru.png")
    plt.cla()
    
    f, ax = plt.subplots(figsize=(11, 15)) 
    heatmap_generated = sns.heatmap(cov_matrix_basic, 
                      square = True,
                      linewidths = .5,
                      cmap = 'coolwarm',
                      cbar_kws = {'shrink': .4, 
                                'ticks' : [-1, -.5, 0, 0.5, 1]},
                      vmin = -1, 
                      vmax = 1,
                      annot = True,
                      annot_kws = {'size': 12})
  
    
    ax.set_yticklabels(cols, rotation = 0)
    ax.set_xticklabels(cols,rotation=25)
    ax.set_ylim(len(cols), -1)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    sns.set_style({'xtick.bottom': True}, {'ytick.left': True})
    plt.savefig("corr_basic.png")
    plt.cla()
    
    
    
    
    
    