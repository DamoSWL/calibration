import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator




np.random.seed(19680801)

if __name__ == '__main__':

    # with np.load('edge_weight_cora.npz') as data:
    #     adj = data['cora_edge']


    # with np.load('pubmed_new_gcn_entropy.npz') as data:
    #     new_gcn_entropy= data['entropy']
 
    # with np.load('pubmed_old_gcn_entropy.npz') as data:
    #     old_gcn_entropy= data['entropy']

    # num_bins = 80

    # plt.figure(figsize=(6, 4))

    # # the histogram of the data
    # n, bins, patches = plt.hist(new_gcn_entropy, num_bins, density=True,color='darkmagenta',alpha=0.6,label='Ours')
    # n, bins, patches = plt.hist(old_gcn_entropy, num_bins, density=True,color='lightskyblue',alpha=0.6, label='GCN')

    # plt.xlabel('Entropy',fontsize=14)
    # plt.ylabel('Density',fontsize=14)
    # plt.title('Histogram of Entropy',fontsize=14)
    # plt.tick_params(labelsize=13)

    # # Tweak spacing to prevent clipping of ylabel
    # plt.legend(fontsize=14)
    # plt.savefig('images/' + 'entropy' + 'pubmed' + '.png' , format='png', dpi=300,
    #                 pad_inches=0, bbox_inches = 'tight')





    plt.figure(figsize=(5,4))
    # plt.rcParams['axes.labelweight'] = 'bold'
    # plt.rcParams["font.weight"] = "bold"

    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    seq = [0,1,2,3]
    conf_1 = [0.0911,0.5403,0.1032,0.2654]
    conf_2 = [0.0654,0.6631,0.0927,0.1788]
    # plt.bar(seq, conf_1, alpha=0.5, width=0.50, color='lightcoral', label='Before')
    plt.bar(seq,conf_2, alpha=0.5, width=0.50, color='dodgerblue', label='After')       
    plt.xlabel('Label', fontsize=14)
    plt.ylabel('Probability', fontsize=14)
    plt.tick_params(labelsize=14)
    # plt.xlim(0, 1.0)
    plt.ylim(0, 1.0)
    #title = 'Uncal. - Cora - 20 - GCN'
    # plt.title(title, fontsize=16, fontweight="bold")
    plt.legend(fontsize=14)
    plt.savefig('images/'   + 'after' + '_' + 'cora'+ '.png' , format='png', dpi=300,
                pad_inches=0.2, bbox_inches = 'tight')
    




    
    # y = [1.41,0.91,1.25,1.31,1.39,1.61,3.21]
    # label = ['cora','citeseer','pubmed','computers','CS','physics','arxiv']

    # x = range(len(y))

    # plt.figure(figsize=(10,6))
    # # plt.style.use('seaborn')

    # plt.bar(x, y, width=0.5, tick_label=label,fc='c')
    
    # plt.xticks(fontsize = 15) 

    # plt.legend(fontsize=15)

    # plt.xlabel('Benchmark',fontsize=15)
    # plt.ylabel('Inference time(s)',fontsize=15)

    # plt.savefig('images/'   + 'scale' + '.png' , format='png', dpi=300,
    #             pad_inches=0.2, bbox_inches = 'tight')
