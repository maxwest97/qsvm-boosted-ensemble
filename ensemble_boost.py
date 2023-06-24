import numpy as np, multiprocessing as mp
from functools import partial
from sklearn.svm import SVC
import os, subprocess, sys
import argparse
from time import time

from c2qa import *  # https://github.com/C2QA/bosonic-qiskit/tree/main
from utils  import *
from circuit import *

np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

pm       = 1
cv       = 0
pca_cv   = 1

parser = argparse.ArgumentParser()
parser.add_argument("--nensemble", type=int, default=1)
parser.add_argument("--nfactory", type=int, default=15)
parser.add_argument("--npca", type=int, default=2)
parser.add_argument("--reduced_ntrain", type=int, default=50000)
parser.add_argument("--ntest", type=int, default=50000)
parser.add_argument("--predict_probs", type=int, default=0)
parser.add_argument("--equal_bins", type=int, default=0)
parser.add_argument("--regularisation", type=float, default=1.0)
parser.add_argument("--offset", type=int, default=0)
parser.add_argument("--train_offset", type=int, default=0)
parser.add_argument("--num_qubits_per_qumode", type=int, default=3)
parser.add_argument("--num_qumodes_per_cut", type=int, default=5)
parser.add_argument("--nqubits", type=int, default=10)
parser.add_argument("--nlayers", type=int, default=13)

args = vars(parser.parse_args())

nensemble             = args["nensemble"]
nlayers               = [ args["nlayers"] ] * nensemble 
nfactory              = [ args["nfactory"] ] * nensemble 
circuit_arch          = [ pca_cv_circuit ] if pca_cv else [ original_circuit ] * len(nfactory)
reduced_ntrain        = args["reduced_ntrain"]
ntest                 = args["ntest"]
npca                  = args["npca"]
predict_probs         = args["predict_probs"]
equal_bins            = args["equal_bins"]
regularisation        = args["regularisation"]
offset                = args["offset"]
train_offset          = args["train_offset"]
num_qubits_per_qumode = args["num_qubits_per_qumode"]
num_qumodes_per_cut   = args["num_qumodes_per_cut"]
nqubits               = [ num_qubits_per_qumode * num_qumodes_per_cut if pca_cv else args["nqubits"] ] * nensemble 
num_bins              = 7
checkpoints           = [[ 1 + i for i in range(nfactory[0]) ] for _ in range(nensemble) ]
nprocesses            = 4
create_gif            = 0
shuffle_data          = 0
narchitectures        = len(nfactory)
ncheckpoints          = len(checkpoints[0])
ntrain                = reduced_ntrain * sum(nfactory)
parallel_prep         = 1
load_test_embed       = 1
save_test_embed       = 1 - load_test_embed
disp                  = 1

if parallel_prep:
    assert nensemble == 1

np.random.seed(959)

# data generated using the Belle-II analysis framework
# https://software.belle2.org/

if train_offset < 180 or reduced_ntrain < 50000:
	train_data_file = './pca_' + str(npca) + '_9mdata.npy'   
	batch = 0

elif train_offset >= 180  and train_offset < 350:
	train_data_file = './pca_' + str(npca) + '_a_85mdata.npy'
	batch = 1

#elif train_offset >= 170 and train_offset < 255:
#	train_data_file = '../../transformed_cv_qsvm_b_85mdata.npy'
#	batch = 2
#
#elif train_offset >= 255:
#	train_data_file = '../../transformed_cv_qsvm_c_85mdata.npy'
#	batch = 3
	
test_data_file  = './pca_' + str(npca) + '_1mdata.npy'

def feature_map(x, circuit, nqubits, pid, disp=0, params=None):
    
	n = x.shape[0]
	states = np.zeros((n, 2 ** nqubits)).astype(np.complex64)
    
	for i in range(pid, n, nprocesses):

		if cv or pca_cv:
			states[i, :] = circuit(x[i], num_qumodes_per_cut, num_qubits_per_qumode, params)
		else:
			states[i, :] = circuit(x[i], params)
	
	return states

if __name__ == '__main__':
    
    print('\n\n *******************************\n')
    print('\n Beginning new run...\n reduced_ntrain = ' + str(reduced_ntrain) + ', ntest = ' + str(ntest))
    print(' Ensemble Size: ' + str(nensemble))
    print(' Probabilities: ' + str(predict_probs))
    print('    Equal Bins: ' + str(equal_bins))
    print('        Offset: ' + str(offset))
    print('Params (q, l, f): ')
	
    for i in range(narchitectures)[:1]:
        print('       (' + str(0 if (cv or pca_cv) else nqubits[i]) + ', ' + str(0 if (cv or pca_cv) else nlayers[i]) + ', ' + str(nfactory[i]) + ')')
	
    params = None if (cv or pca_cv) else np.random.uniform(0, 2 * np.pi, (nlayers[i], 2, nqubits[i]))
    pool   = mp.Pool(processes=nprocesses)
	
    train_correct = 0
    train_acc = []
    votes = np.zeros((narchitectures, ncheckpoints, ntest))
    alpha_history = np.zeros((narchitectures, ncheckpoints))
    test, test_flavours = read_data(test_data_file, ntest * offset, ntest * (offset + 1), pm=pm, nn=1, qsvm=0)
    test *= 0.1 
    qr_index = 0

    if cv:
        
        if load_test_embed:
            
            embedding_test = np.load('test_embed_' + str(num_qubits_per_qumode) + '_' + str(num_qumodes_per_cut) + '.npy')

        else:
            
            print(test.shape[1], num_qumodes_per_cut, num_qubits_per_qumode, test.shape[1] // num_qumodes_per_cut, flush=True)
            
            embedding_test = np.array([ sum(pool.map(partial(feature_map, test[:,num_qumodes_per_cut*i:num_qumodes_per_cut*(i+1)],  cv_circuit, num_qubits_per_qumode * num_qumodes_per_cut, disp=disp, params=params), range(nprocesses))) for i in range(test.shape[1] // num_qumodes_per_cut) ])

            if save_test_embed:

                np.save('test_embed_' + str(num_qubits_per_qumode) + '_' + str(num_qumodes_per_cut), embedding_test)
                print('test_embed', embedding_test.shape)
                exit()
    elif pca_cv:

        if load_test_embed:

            embedding_test = np.load('test_embed.npy')

        else:

            print(test.shape[1], flush=True)
            t=time()

            embedding_test = feature_map(test, pca_cv_circuit, num_qubits_per_qumode * num_qumodes_per_cut, 0, params=params)
            print('time: ', time() - t, flush=True)
            #embedding_test = np.sum(np.array(pool.map(partial(feature_map, test, pca_cv_circuit, num_qubits_per_qumode * num_qumodes_per_cut, params=params), range(nprocesses))))

            if save_test_embed:

                np.save('test_embed', embedding_test)
                print('test_embed', embedding_test.shape)
                exit()
    else:

        if load_test_embed:
            
            embedding_test = np.load('test_embed_' + str(nqubits) + '_' + str(nlayers) + '.npy')

        else:
            
            print(test.shape[1], nqubits, nlayers, flush=True)
            
            embedding_test = sum(pool.map(partial(feature_map, test, circuit_arch[0], nqubits[0], disp=disp, params=params), range(nprocesses)))

            if save_test_embed:

                np.save('test_embed_' + str(nqubits) + '_' + str(nlayers), embedding_test)
                print('test_embed', embedding_test.shape)
                exit()

    for i in range(narchitectures):
   
        print(train_offset, 'init: ', reduced_ntrain * (train_offset + i - 180 * batch), 'final: ', reduced_ntrain * (train_offset + i + 1 - 180 * batch))
        train, train_flavours = read_data(train_data_file, reduced_ntrain * (train_offset + i - 180 * batch), reduced_ntrain * (train_offset + i + 1 - 180 * batch), pm=pm, nn=1, qsvm=0)
        train *= 0.1
        
        if not(i):
            
            print(train.shape, train_flavours.shape, test.shape, test_flavours.shape, flush=True)
        
        if cv:
			
            mtrain = np.ones((train.shape[0], train.shape[0]))
            mtest  = np.ones((test.shape[0],  train.shape[0]))

            for j in range(train.shape[1] // num_qumodes_per_cut):
				
                embedding_train = sum(pool.map(partial(feature_map, train[:,num_qumodes_per_cut*j:num_qumodes_per_cut*(j+1)], cv_circuit, num_qubits_per_qumode * num_qumodes_per_cut, disp=disp, params=params), range(nprocesses)))

                mtrain *=  np.abs(np.conj(embedding_train)   @ embedding_train.T) ** 2 
                mtest  *=  np.abs(np.conj(embedding_test[j]) @ embedding_train.T) ** 2 
        elif pca_cv:

            embedding_train = feature_map(train, pca_cv_circuit, nqubits[i], 0, params=params)
            mtrain = np.abs(np.conj(embedding_train) @ embedding_train.T) ** 2
            mtest  = np.abs(np.conj(embedding_test)  @ embedding_train.T) ** 2

        else:		

            embedding_train = sum(pool.map(partial(feature_map, train, circuit_arch[i], nqubits[i], disp=disp, params=params), range(nprocesses)))
		
            mtrain = np.abs(np.conj(embedding_train) @ embedding_train.T) ** 2
            mtest  = np.abs(np.conj(embedding_test)  @ embedding_train.T) ** 2
		
        ckpt = 0
        done = 0
		
        weights = np.ones(reduced_ntrain)
        svm = SVC(kernel="precomputed", C=regularisation, probability=predict_probs)
		
        for j in range(nfactory[i]):
            
            if done:
                break
            
            if (j >= checkpoints[i][ckpt]) and (i == narchitectures - 1):
                train_acc.append(1 - mistakes[0].shape[0] / reduced_ntrain)
            if j >= checkpoints[i][ckpt]:
                votes[i,ckpt+1] = votes[i,ckpt] 
                alpha_history[i,ckpt+1] = alpha_history[i,ckpt] 
                ckpt += 1
            
            svm.fit(mtrain, train_flavours, sample_weight=weights)
            mistakes = np.where(svm.predict(mtrain) != train_flavours)
            eps = np.sum(weights[mistakes]) / np.sum(weights)
            
            if not(eps):
                eps = 0.0001 / np.sum(weights)
                done = 1
            
            alpha = np.log((1-eps)/eps)
            weights[mistakes] *= (np.e ** alpha)
            #print(f'train acc: {1 - mistakes[0].shape[0] / reduced_ntrain:.3f}, eps: {eps:.3f}, alpha: {alpha:.3f} ')
			
            if predict_probs:
                votes[i,ckpt] += alpha * svm.predict_proba(mtest)[:,1]   
            else:
                votes[i,ckpt] += alpha * svm.predict(mtest)
            #print(i,j,ckpt, votes[i,ckpt], flush=True)

            alpha_history[i,ckpt] += alpha

    train_acc.append(1 - mistakes[0].shape[0] / reduced_ntrain)
    
    print(alpha_history.shape)
    
    if parallel_prep and pca_cv:
        np.save('qr/ah_' + str(train_offset) + '_' + str(reduced_ntrain) + '_' + str(ntest) + '_' + str(predict_probs) + '_' + str(equal_bins) + '_' + str(regularisation) + '_' + str(num_qubits_per_qumode) + '_' + str(num_qumodes_per_cut) + '.npy', alpha_history)
    elif parallel_prep and not(cv):
        np.save('qr/ah_' + str(train_offset) + '_' + str(reduced_ntrain) + '_' + str(ntest) + '_' + str(predict_probs) + '_' + str(equal_bins) + '_' + str(regularisation) + '_' + str(nlayers) + '_' + str(nqubits) + '.npy', alpha_history)
    
    test_acc = []
    tag_eff  = []

    #for i in itertools.product(*[ range(ncheckpoints) for _ in range(narchitectures)]):
    for i in [ [ n ] * len(nfactory) for n in range(ncheckpoints) ]:
        
        print('i', i)
        ckpt_vote = np.zeros(ntest)
        voting_population = 0

        for j in range(len(i)):
            ckpt_vote += votes[j, i[j]]
            voting_population += alpha_history[j][i[j]] #checkpoints[j][i[j]] * alpha_history[j][i[j]]

        if not(voting_population):
            break

        qr = ckpt_vote / voting_population if pm else -1 + 2 * ckpt_vote / voting_population
        
        if parallel_prep and pca_cv:
            np.save('qr/cv_' + str(train_offset) + '_' + str(qr_index) + '_' + str(reduced_ntrain) + '_' + str(ntest) + '_' + str(predict_probs) + '_' + str(equal_bins) + '_' + str(regularisation) + '_' + str(num_qubits_per_qumode) + '_' + str(num_qumodes_per_cut) + '.npy', ckpt_vote)
            qr_index += 1
        elif parallel_prep and not(cv):
            np.save('qr/' + str(train_offset) + '_' + str(qr_index) + '_' + str(reduced_ntrain) + '_' + str(ntest) + '_' + str(predict_probs) + '_' + str(equal_bins) + '_' + str(regularisation) + '_' + str(nlayers) + '_' + str(nqubits) + '.npy', ckpt_vote)
            qr_index += 1

        r  = np.abs(qr)
        r[r<0]=0
        r[r>1]=1
        
        if i[0] and equal_bins:
            sorted_r = np.sort(r)
            cutoffs = np.array([-0.01, *[ sorted_r[i * len(sorted_r) // num_bins] for i in range(1, num_bins)], 1.01 ])
            r_thresholds = np.array([ np.array([cutoffs[i], cutoffs[i+1]]) for i in range(cutoffs.shape[0] - 1) ])
            print(r_thresholds)
        else:
            r_thresholds = np.array([ np.array([-0.01, 0.1]), np.array([0.1, 0.25]), np.array([0.25, 0.5]), np.array([0.5, 0.625]), np.array([0.625, 0.75]), np.array([0.75, 0.875]), np.array([0.875, 1.01])   ])
        
        bins = [ np.where((r > x[0]) & (r <= x[1]))[0] for x in r_thresholds  ]
        fractions = np.array([ b.shape[0] / r.shape[0] for b in bins ])
        if not(np.sum(fractions)):
            break

        if pm:
            test_acc.append(sum((ckpt_vote > 0).astype(int) == (test_flavours > 0).astype(int)) / ntest)
        else:
            test_acc.append(sum((ckpt_vote > voting_population / 2).astype(int)==test_flavours) / ntest)

        print('Checkpoint: ', [checkpoints[j][i[j]] for j in range(len(i))], 'Accuracy: ', round(test_acc[-1], 3))

        assert np.abs(np.sum(fractions)-1) < 1e-3, (fractions, np.amin(fractions), np.amax(fractions), np.sum(fractions), np.amin(r), np.amax(r), r, qr)
        if pm:
            wrong = np.array([ np.sum((ckpt_vote[b] > 0).astype(int) != (test_flavours[b] > 0).astype(int)) / b.shape[0] if b.shape[0] else 0 for b in bins ])
        else:
            wrong = np.array([ np.sum((ckpt_vote[b] > voting_population / 2).astype(int) != test_flavours[b]) / b.shape[0] if b.shape[0] else 0 for b in bins ])
        
        tag_eff.append(np.dot(fractions, (1 - 2 * wrong)**2))
        
        print('fraction in bins:', fractions, '\n            accs:', 1 - wrong, '\n         tag_eff: ', tag_eff[-1])

    if cv or pca_cv:
        print([nfactory[0], reduced_ntrain, ntest, predict_probs, equal_bins, nensemble, regularisation, num_qubits_per_qumode, num_qumodes_per_cut], ':', tag_eff)
    else:
        print([nqubits[0], nlayers[0], nfactory[0], reduced_ntrain, ntest, predict_probs, equal_bins, nensemble, regularisation], ':', tag_eff)


    pool.close()
    pool.join() 

