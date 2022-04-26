import pickle as pkl
import numpy as np
from timegan2 import *
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == "__main__":
    
    
    generated = []
    
    for i in range(5):
        file_path = 'transformer'+str(i+1)+'/generated.pkl'
        file = open(file_path,'rb')
        arr = pkl.load(file)
        file.close()
        generated.append(arr)
    
    all_generated = np.concatenate(generated)
    all_generated = np.reshape(all_generated ,(-1,7))


    X = RealDataset(os.path.join("data", "features.csv"), dt.datetime(1995, 1, 3), dt.datetime(2019, 12, 31))
    loader = DataLoader(X, batch_size = len(X))
    real_data = next(iter(loader)).numpy()

    
    all_generated =  all_generated*(X.max[None,:] - X.min[None,:])+X.min[None,:] 
    
    real_data = np.reshape(real_data ,(-1,7))
    real_data = real_data*(X.max[None,:]  - X.min[None,:]) +X.min[None,:] 
    
    ##### calc means ######
    avg_real = np.mean(real_data,0)
    avg_generated = np.mean(all_generated,0)
    
    print(avg_real)
    print(avg_generated)
    print()
    print(np.abs(avg_real - avg_generated)/avg_real*100)
    print()
    
    
    ##### calc var #####
    var_real = np.std(real_data,0, ddof=1)
    var_generated = np.std(all_generated,0, ddof=1)
    
    print(var_real)
    print()
    print(var_generated)
    print()
    print(np.abs(var_real - var_generated)/var_real*100)
    
    #### calc cov ######
    cov_matrix_real = np.corrcoef(real_data, rowvar=False)
    cov_matrix_generated = np.corrcoef(all_generated, rowvar=False)
    
    
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
    
    
    
    
    