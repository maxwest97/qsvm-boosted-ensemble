import multiprocessing as mp
import numpy as np
from functools import partial
import matplotlib as mpl
from sklearn.svm import SVC
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from time import time
import pickle, sys, torch

def calc_tag_eff(pred, flavours, num_bins, equal_bins, linear):
    
    if len(pred.shape) > 1:
        if pred.shape[1] == 2:
            pred  = pred[:,1]
        else: 
            pred  = pred[:,0]
    
    if linear:
        pred = pred - np.amin(pred)
        pred = pred / np.amax(pred)
    
    qr = -1 + 2 * pred
    #print(np.amin(qr), np.max(qr))
    
    r  = np.abs(qr)
    r[r==1] += 0.0001 * (np.random.random(r[r==1].shape[0]) - 1/2)
    
    if equal_bins:
        sorted_r = np.sort(r)
        cutoffs = np.array([-0.001, *[ sorted_r[i * len(sorted_r) // num_bins] for i in range(1, num_bins)], 1.001 ])
        r_thresholds = np.array([ np.array([cutoffs[i], cutoffs[i+1]]) for i in range(cutoffs.shape[0] - 1) ])
    
    else:
        r_thresholds = np.array([ np.array([-0.001, 0.1]), np.array([0.1, 0.25]), np.array([0.25, 0.5]), np.array([0.5, 0.625]), np.array([0.625, 0.75]), np.array([0.75, 0.875]), np.array([0.875, 1.001])   ])
        
    bins = [ np.where((r > x[0]) & (r <= x[1]))[0] for x in r_thresholds  ]
    fractions = np.array([ b.shape[0] / r.shape[0] for b in bins ])
     
    assert np.abs(np.sum(fractions)-1) < 1e-3, (fractions, np.amin(fractions), np.amax(fractions), np.sum(fractions), r, qr)
    
    wrong = np.array([ np.sum((qr[b] > 0).astype(int) != (flavours[b].detach().numpy() > 0).astype(int)) / b.shape[0] if b.shape[0] else 0 for b in bins ])
    tag_eff = np.dot(fractions, (1 - 2 * wrong)**2)
    acc = 1 - np.dot(fractions, wrong) ; print('acc: ', acc)
    
    return tag_eff, r, wrong, bins, r_thresholds


def combine(test_flavours, nensemble, nfactory, reduced_ntrain, ntest, predict_probs, equal_bins, regularisation, num_qubits_per_qumode, num_qumodes_per_cut, num_bins, linear):
    
	tag_effs = []
	total_votes = np.zeros((nfactory, ntest))
	
	for i in range(nfactory):
	    
		votes = np.zeros(ntest)
		voting_population = 0

		for j in range(nensemble):

			try:
				votes += np.load('qr/cv_' + str(j) + '_' + str(i) + '_' + str(reduced_ntrain) + '_' + str(ntest) + '_' + str(predict_probs) + '_' + str(equal_bins) + '_' + str(regularisation) + '_' + str(num_qubits_per_qumode) + '_' + str(num_qumodes_per_cut) + '.npy')
				voting_population += np.load('qr/ah_'    + str(j) + '_' + str(reduced_ntrain) + '_' + str(ntest) + '_' + str(predict_probs) + '_' + str(equal_bins) + '_' + str(regularisation) + '_' + str(num_qubits_per_qumode) + '_' + str(num_qumodes_per_cut) + '.npy')[0,i]

			except FileNotFoundError:
				print('qr/cv_' + str(j) + '_' + str(i) + '_' + str(reduced_ntrain) + '_' + str(ntest) + '_' + str(predict_probs) + '_' + str(equal_bins) + '_' + str(regularisation) + '_[' + str(num_qubits_per_qumode) + ']_[' + str(num_qumodes_per_cut) + '].npy')
				print('.')
				pass
		
		if not(voting_population):
			tag_effs.append(0)

		else:
			votes /= voting_population
			print(votes)
			total_votes[i] = votes
			tag_effs.append(calc_tag_eff((votes+1)/2, torch.tensor(test_flavours), num_bins, equal_bins, linear)[0])

	np.save('total_votes.npy', total_votes)
	np.save('test_flavours_50k.npy', test_flavours)
	return tag_effs

def rw_plot(r, wrong, bins, r_thresholds):

    r_ave   = [ np.sum(r[bins[i]]) / bins[i].shape[0] for i in range(len(bins)) ]
    results = [ 1 - 2*wrong[i] for i in range(len(bins)) ]
    
    for i in range(len(bins)): 
        print(r_thresholds[i], f'ave: {np.sum(r[bins[i]]) / bins[i].shape[0]:.3f}, result:  {1 - 2*wrong[i]:.3f}, eps: {bins[i].shape[0] / r.shape[0]:.3f}')
    
    #plt.scatter(range(len(bins)), r_ave, label=r'$\langle r_{i} \rangle $')
    #plt.scatter(range(len(bins)), results, label=r'$1-2w_{i}$')
    #
    #plt.xlabel('Bin Number')
    #plt.title(r'Average value of $r$ and $1-2w$ for each bin')
    #plt.legend()
    #plt.show()
    #plt.savefig('rvw/' + str(num_bins) + '.png')
    #plt.close()
    return r_ave, results

def read_data(file_name, start, end, ncharge=None, npi=None, specific_ncharge=None, include_ncharge=0, pm=0, belle2=0, nn=0, qsvm=0):
    
    #data = np.nan_to_num(np.load(file_name)[start:end])
    data = np.load(file_name)[start:end]
    
    # kill num charge
    #assert end <= data.shape[0], f'{end, data.shape[0]}' 
    if nn:
        
        X = data[:,2:]
        y = ((data[:,1] + 1) / 2).astype(int) if not(pm) else data[:,1].astype(int)
        
    elif qsvm:
        
        X = np.hstack((data[:,2:37], data[:,67:102]))
        y = ((data[:,1] + 1) / 2).astype(int) if not(pm) else data[:,1].astype(int)
        
    elif ncharge is not None:
        
        X = np.hstack((data[:,4-include_ncharge:4+(9+6*belle2)*ncharge], data[:,4+8*(9+6*belle2):4+8*(9+6*belle2)+4*npi]))
        y = ((data[:,1] + 1) / 2).astype(int) if not(pm) else data[:,1].astype(int)

    elif specific_ncharge is not None:

        #data = data[start:end,:]
        
        if specific_ncharge < 8:

            #X = data[data[:,3]==specific_ncharge,4:4+9*specific_ncharge] 
            X = np.hstack((data[data[:,3]==specific_ncharge,4:4+9*specific_ncharge], data[data[:,3]==specific_ncharge,76:76+4*npi])) 
            y = ((data[data[:,3]==specific_ncharge, 1] + 1) / 2).astype(int) if not(pm) else (data[data[:,3]==specific_ncharge, 1]).astype(int)
        else:
            
            X = data[data[:,3]>=8,4:4+9*8]
            #X.append(np.hstack((data[data[:,3]>=8,4:4+9*8], data[data[:,3]>=8,76:92])))
            y = ((data[data[:,3]>=8,1] + 1) / 2).astype(int) if not(pm) else (data[data[:,3]>=8,1]).astype(int)
    
    return X, y

def accuracy(classifier, X, Y_target):
    return sum(classifier.predict(X) == Y_target) / len(Y_target)

def multithread_acc(classifier, data, labels, num_threads, i):
	
    box      = int(len(data)/num_threads)
    X        =   data[i*box:(i+1)*box if i < num_threads - 1 else data.shape[0], :]
    Y_target = labels[i*box:(i+1)*box if i < num_threads - 1 else data.shape[0]]
    
    return sum(classifier.predict(X) == Y_target) 

def scatter_data(data, data_flavours, mistakes, generation):

    wrong       = data[mistakes]
    wrong_fl    = data_flavours[mistakes]
    correct     = np.delete(data, mistakes, axis=0)
    correct_fl  = np.delete(data_flavours, mistakes, axis=0)
    
    f, ax = plt.subplots(1,2, figsize=(12.5,5.3), gridspec_kw={'width_ratios': [1, 1]})
    
    ax[0].annotate(f'Generation {generation}', xy=(0.435, 0.85), xycoords='figure fraction',xytext=(20, 20), textcoords='offset points', fontsize=15)
	
    [ ax[x[0]].title.set_text(x[1]) for x in [(0,'Correct'), (1,'Wrong'), ]  ]
    [ ax[i].set_xlabel(r'$p_{c_0}$') for i in range(2) ]
    [ ax[i].set_ylabel(r'$p_{\pi_0}$') for i in range(2) ]
    [ ax[i].set_xlim([-2.5, 2.5]) for i in range(2) ]
    [ ax[i].set_ylim([   0, 2.5]) for i in range(2) ]
	
    size = 0.5

    ax[0].scatter(correct[np.where(correct_fl),0]   *  correct[np.where(correct_fl),3],   correct[np.where(correct_fl),9], color='b', s=size)
    ax[0].scatter(correct[np.where(1-correct_fl),0] *  correct[np.where(1-correct_fl),3], correct[np.where(1-correct_fl),9], color='r', s=size)
    ax[1].scatter(wrong[np.where(wrong_fl),0]   *  wrong[np.where(wrong_fl),3],   wrong[np.where(wrong_fl),9], color='b', s=size)
    ax[1].scatter(wrong[np.where(1-wrong_fl),0] *  wrong[np.where(1-wrong_fl),3], wrong[np.where(1-wrong_fl),9], color='r', s=size)
    #ax[0,2].imshow(np.zeros((28,28)), cmap=colour)
	
    plt.savefig('mplots/gen_' + str(generation) + '.png', dpi=1200)
    #plt.show()
    plt.close()
    #exit()
    return 

