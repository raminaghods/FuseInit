# -*- coding: utf-8 -*-
"""
Code for paper:

Ghods, Ramina and Lan, Andrew S and Goldstein, Tom and Studer, Christoph, ``MSE-optimal
neural network initialization via layer fusion", 2020 54th Annual Conference on
Information Sciences and Systems (CISS)

(c) 2020 raminaghods (rghods@cs.cmu.edu)

fuses multiple layers from of the network one at a time, you can fuse any layer

data is from:
 D. Anguita, A. Ghio, L. Oneto, X. Parra, and J. Reyes-Ortiz, “A public
domain dataset for human activity recognition using smartphones,” in
21th European Symposium on Artificial Neural Networks, Computational
Intelligence and Machine Learning (ESANN). CIACO, Apr. 2013, pp.
437–442.

you need to download the data and put it in a folder data
"""

This is the code to collapse any of the 1dconv layers of the 4 1dconv layered network created by HAR-CNN.py
the data is called HAR: Human Activity Recognition

"""

import sys
import numpy as np
import os
#sys.path.append("../..")
from utilities import read_data
from utilities import standardize
from utilities import one_hot
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from trntst_HAR import trntst_HAR
import seaborn as sb
import pickle
from Plot_FuseNet import Plot_FuseNet
#get_ipython().run_line_magic('matplotlib', 'Automatic')

#if (os.path.exists('checkpoints-cnn') == False):
#    get_ipython().system('mkdir checkpoints-cnn')


#%% Hyperparameters
#np.random.seed(400)
batch_size = 1000      # Batch size
seq_len = 128          # Number of steps
lrn_rate = 0.0001
epochs = 1000
krnl_sz = 3 # kernel size, i.e. length of filter, only odd numbers for now are possible
krnl_sz_Bsg = 3

L = 2
K = 1
act_func = 'relu'
n_classes = 6
n_channels = 9
N = np.array([36,18,72,144]) # number of output channels of each layer
Spool = np.array([2,2,2,2]) # strides of each maxpool layer
Sconv = np.array([1,1,1,1]) # strides of each convolution layer

num_trials = 10
he_init = True
if(he_init):
    from build_234layer_1dconv_graph_he import build_234layer_1dconv_graph
else:
    from build_234layer_1dconv_graph_rnd import build_234layer_1dconv_graph


par = {'batch_size':batch_size,'seq_len':seq_len,'lrn_rate':lrn_rate,'epochs':epochs,
         'krnl_sz':krnl_sz,'krnl_sz_Bsg':krnl_sz_Bsg,'L':L,'K':K,'n_classes':n_classes,'n_channels':n_channels
          ,'n_outchannel':N,'Spool':Spool,'Sconv':Sconv,'num_trials':num_trials,'act_func':act_func}
#%% Prepare data

X_train, labels_train, list_ch_train = read_data(data_path="data/", split="train") # train
X_test, labels_test, list_ch_test = read_data(data_path="data/", split="test") # test

assert list_ch_train == list_ch_test, "Mistmatch in channels!"
# Normalize?
X_train, X_test = standardize(X_train, X_test)

# Train/Validation Split
X_tr, X_vld, lab_tr, lab_vld = train_test_split(X_train, labels_train,
                                                stratify = labels_train, random_state = 123)

# One-hot encoding:
y_tr = one_hot(lab_tr)
y_vld = one_hot(lab_vld)
y_test = one_hot(labels_test)

X_tr = X_tr[:,:,0:n_channels]
X_test = X_test[:,:,0:n_channels]
X_vld = X_vld[:,:,0:n_channels]

#%% initialization:
[graph,inputs_,labels_,keep_prob_,learning_rate_plc_holdr,cost, optimizer, accuracy,z2t,z3t,z4t,a1t,a2t,Wdt,bdt] = build_234layer_1dconv_graph(seq_len,
                     n_channels,n_classes,L,N,krnl_sz,Sconv,Spool)
rnd2ttrr = trntst_HAR(graph,inputs_,labels_,
                      keep_prob_,learning_rate_plc_holdr,cost, optimizer, accuracy,z2t,z3t,z4t,a1t,a2t,Wdt,bdt,epochs,
                      X_tr,y_tr,X_vld,y_vld,X_test,y_test,batch_size,lrn_rate)

[tr_acc,vld_acc,tr_loss,vld_loss] = rnd2ttrr[0:4]

train_acc_2rnd = np.zeros((tr_acc.shape[0],num_trials))
validation_acc_2rnd = np.zeros((vld_acc.shape[0],num_trials))
train_loss_2rnd = np.zeros((tr_loss.shape[0],num_trials))
validation_loss_2rnd = np.zeros((vld_loss.shape[0],num_trials))
test_acc_2rnd = np.zeros((1,num_trials))
train_acc_1Bsg = np.zeros((tr_acc.shape[0],num_trials))
validation_acc_1Bsg = np.zeros((vld_acc.shape[0],num_trials))
train_loss_1Bsg = np.zeros((tr_loss.shape[0],num_trials))
validation_loss_1Bsg = np.zeros((vld_loss.shape[0],num_trials))
test_acc_1Bsg = np.zeros((1,num_trials))
train_acc_1rnd = np.zeros((tr_acc.shape[0],num_trials))
validation_acc_1rnd = np.zeros((vld_acc.shape[0],num_trials))
train_loss_1rnd = np.zeros((tr_loss.shape[0],num_trials))
validation_loss_1rnd = np.zeros((vld_loss.shape[0],num_trials))
test_acc_1rnd = np.zeros((1,num_trials))

rnd2 = []
rnd1 = []
Bsg1 = []
for ttrr in range(0,num_trials):
    #%% create L-layer 1dconv with random initialization:

    if(ttrr != 0):
        [graph,inputs_,labels_,keep_prob_,learning_rate_plc_holdr,cost, optimizer, accuracy,z2,z3,z4,a1,a2,Wd,bd] = build_234layer_1dconv_graph(
                seq_len, n_channels,n_classes,L,N,krnl_sz,Sconv,Spool)
        rnd2ttrr = trntst_HAR(graph,inputs_,labels_,keep_prob_,learning_rate_plc_holdr,cost, optimizer, accuracy,z2,z3,z4,a1,a2,Wd,bd,epochs,
                             X_tr,y_tr,X_vld,y_vld,X_test,y_test,batch_size,lrn_rate)

    rnd2.append(rnd2ttrr)
    [train_acc_2rnd[:,ttrr],validation_acc_2rnd[:,ttrr],train_loss_2rnd[:,ttrr],validation_loss_2rnd[:,ttrr],test_acc_2rnd[:,ttrr]] = rnd2ttrr[0:5]
    [z2_opt,z3_opt,z4_opt,a1_opt,a2_opt] = rnd2ttrr[5:10]
    z = rnd2ttrr[5:8]
    a = rnd2ttrr[8:10]
    Wdo = rnd2ttrr[10]
    bdo = rnd2ttrr[11]
    n_tr = z2_opt.shape[0]
    a.insert(0,X_tr[0:n_tr,:,:])


    #%% compute Bussgang initialization:
    z2_Bsg = z[K-1]
    a0 = a[K-1]
    allN = np.hstack((n_channels,N))
    h_tilde = np.zeros((krnl_sz_Bsg,allN[K-1], allN[K+1]),dtype=np.float32)
    stilde = Sconv[K-1]*Spool[K-1]*Sconv[K]
    v_p = z2_Bsg - np.mean(z2_Bsg,axis=0)
    L0 = a0.shape[1]
    epsilon = 1e-3
    for mm in range(0,allN[K-1]): # in channel idx of 1st layer
        u_tmp = np.transpose(a0[0:n_tr,:,mm] - np.mean(a0[0:n_tr,:,mm]))
        u = np.vstack((np.zeros((int((krnl_sz_Bsg-1)/2),n_tr)),np.flipud(u_tmp),np.zeros((int((krnl_sz_Bsg-1)/2),n_tr))))
        for nn in range(0,allN[K+1]): # output channel idx of 2nd layer N2
            v = np.transpose(v_p[:,:,nn])
#            v_upsampled = np.insert(vtmp, slice(1, None), 0,axis=0)
#            v_tilde = np.vstack((np.zeros((int((krnl_sz_Bsg-1)/2),n_tr)),v_upsampled,np.zeros((int((krnl_sz_Bsg-1)/2+1),n_tr))))
            V = np.zeros((krnl_sz_Bsg,1))
            U = np.zeros((krnl_sz_Bsg,krnl_sz_Bsg))
            for ii in range(0,L0,stilde):#range(1,seq_len,2):
                u_sub = u[L0-ii-1:L0-ii+krnl_sz_Bsg-1,:]
                v_sub = v[int(ii/stilde),:]
                U += np.matmul(u_sub,np.transpose(u_sub))/n_tr
                V += np.reshape(np.mean(v_sub*u_sub,axis=1),[-1,1]) # check this
            Uinv = np.linalg.inv(U+epsilon*np.eye(krnl_sz_Bsg,dtype=np.float32))
            h_tilde[:,mm,nn] = np.squeeze(np.matmul(Uinv,V))

    remainNodes_idx = np.array([True,True,True,True]) # index of those nodes that remain after fusion
    remainNodes_idx[K-1] = False
    rndinitL_1 = remainNodes_idx
    Sconv_Bsg = np.hstack((Sconv[remainNodes_idx],1))
    Sconv_Bsg[K-1] = Sconv[K-1]*Spool[K-1]*Sconv[K]
    Spool_Bsg = np.hstack((Spool[remainNodes_idx],1))
    N_Bsg = np.hstack((N[remainNodes_idx],1))

    hconvb_tilde = np.zeros((int(L0/stilde),allN[K+1]),dtype=np.float32)
    a0bar = np.mean(a0,axis=0)
    for nnn in range(0,allN[K+1]):
        convtmp = np.zeros((L0),dtype=np.float32)
        for mmm in range(0,allN[K-1]):
            convtmp += np.convolve(h_tilde[:,mmm,nnn],a0bar[:,mmm],mode='same')
        hconvb_tilde[:,nnn] = convtmp[0:L0:stilde]
    b_tilde = np.mean(z2_Bsg,axis=0) - hconvb_tilde


    #%% create (L-1)-layer 1dconv with Bussgang initialization:


    [graph,inputs_,labels_,keep_prob_,learning_rate_plc_holdr,cost, optimizer, accuracy,z2,z3,z4,a1,a2,Wdt,bdt] = build_234layer_1dconv_graph(
                            seq_len, n_channels,n_classes,L-1,N_Bsg,krnl_sz_Bsg,Sconv_Bsg,Spool_Bsg,h_tilde,b_tilde,rndinitL_1,False,Wdo,bdo)
    Bsg1ttrr = trntst_HAR(graph,inputs_,labels_,keep_prob_,learning_rate_plc_holdr,
                          cost, optimizer, accuracy,z2,z3,z4,a1,a2,Wdt,bdt,epochs,X_tr,y_tr,X_vld,y_vld,X_test,y_test,batch_size,lrn_rate)
    Bsg1.append(Bsg1ttrr)
    [train_acc_1Bsg[:,ttrr],validation_acc_1Bsg[:,ttrr],train_loss_1Bsg[:,ttrr],validation_loss_1Bsg[:,ttrr],test_acc_1Bsg[:,ttrr]] = Bsg1ttrr[0:5]

    #%% create 1-layer 1dconv with random initialization:

    [graph,inputs_,labels_,keep_prob_,learning_rate_plc_holdr,cost, optimizer, accuracy,z2,z3,z4,a1,a2,Wdt,bdt] = build_234layer_1dconv_graph(
                            seq_len, n_channels,n_classes,L-1,N_Bsg,krnl_sz_Bsg,Sconv_Bsg,Spool_Bsg)

    rnd1ttrr = trntst_HAR(graph,inputs_,labels_,keep_prob_,learning_rate_plc_holdr,
                          cost, optimizer, accuracy,z2,z3,z4,a1,a2,Wdt,bdt,epochs,X_tr,y_tr,X_vld,y_vld,X_test,y_test,batch_size,lrn_rate)

    rnd1.append(rnd1ttrr)
    [train_acc_1rnd[:,ttrr],validation_acc_1rnd[:,ttrr],train_loss_1rnd[:,ttrr],validation_loss_1rnd[:,ttrr],test_acc_1rnd[:,ttrr]] = rnd1ttrr[0:5]


#%% Save Results

savepath = 'results'
filename = ('numtrials'+str(num_trials)+'_epochs'+str(epochs)+
'_batchsize'+str(batch_size)+'_seqlen'+str(seq_len)+'_lrn_rate'+str(lrn_rate)
+'_krnlsz'+str(krnl_sz)+'_krnlszBsg'+str(krnl_sz_Bsg)+'_L'+str(L)+'_K'+str(K)
+'_actfunc-'+act_func+'_nInchannels'+str(n_channels)
+'_noutchannels'+str(N[0])+'-'+str(N[1])+'-'+str(N[2])+'-'+str(N[3])
#+'_Spool'+str(Spool)+'_Sconv'+str(Sconv)
+'he-init'+str(he_init)
+'.pkl')

with open(os.path.join(savepath,filename),'wb') as f:
    pickle.dump([train_loss_2rnd,train_loss_1rnd,train_loss_1Bsg,validation_loss_2rnd,validation_loss_1rnd,validation_loss_1Bsg,train_acc_2rnd
                 ,train_acc_1rnd,train_acc_1Bsg,validation_acc_2rnd,validation_acc_1rnd,validation_acc_1Bsg,test_acc_2rnd,test_acc_1rnd,test_acc_1Bsg],f)

#%%
# Plot training and test loss

#%% plotting result

Plot_FuseNet(L,1,train_loss_2rnd,train_loss_1rnd,train_loss_1Bsg,validation_loss_2rnd,validation_loss_1rnd,validation_loss_1Bsg,train_acc_2rnd
                 ,train_acc_1rnd,train_acc_1Bsg,validation_acc_2rnd,validation_acc_1rnd,validation_acc_1Bsg,test_acc_2rnd,test_acc_1rnd,test_acc_1Bsg,epochs)
