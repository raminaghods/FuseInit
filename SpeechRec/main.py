# -*- coding: utf-8 -*-
"""
Code for paper:

Ghods, Ramina and Lan, Andrew S and Goldstein, Tom and Studer, Christoph, ``MSE-optimal
neural network initialization via layer fusion", 2020 54th Annual Conference on
Information Sciences and Systems (CISS)

(c) 2020 raminaghods (rghods@cs.cmu.edu)

fuses multiple convolutional layers from last layer of the network one at a time

data and structure for data processing from:
P. Warden, “Speech commands: A dataset for limited-vocabulary speech
recognition,” arXiv preprint: 1804.03209, Apr. 2018.

code requires a GPU node

you need to download the data for however many classes you want to train and run LoadandProcessData.py to
create the data arrays first
"""

import keras
import keras.backend as K
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
K.set_session(session)
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0" #use gpu 1 or 0
from matplotlib import pyplot
from scipy.misc import toimage
from keras.datasets import fashion_mnist
from keras.models import Sequential, Model
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Input
from keras import regularizers
from keras.callbacks import LearningRateScheduler
import numpy as np
from matplotlib import pyplot as plt
import pickle
from Plot_MultiFuseNet import Plot_MultiFuseNet
from get_train_test import get_train_test
from keras.utils import to_categorical
import sys
from utilities import standardize
from sklearn.model_selection import train_test_split


#%% parameters
n_clayers = 4
num_Fuse = 1
num_trials = 1
num_epochs = 2
weight_decay = 1e-4
batch_size = 64
k = 3 # kernel size, i.e. length of filter, only odd numbers for now are possible
n_channels = np.array([32,32,64,64,64,64])
N = n_channels
n_Inchannels = 11

Sconv = np.array([1,1,1,1,1,1]) # strides of each convolution layer
Spool = np.array([1,2,1,2,1,2])
k_Bsg = 21


he_init = True
if(he_init):
    from Create_he_Net_initializeALL import Create_rnd_Net
else:
    from Create_rnd_Net_initializeALL import Create_rnd_Net

#%%initialize
trn_acc_2rnd = np.zeros((num_epochs,num_trials))
vld_acc_2rnd = np.zeros((num_epochs,num_trials))
trn_acc_1Bsg = np.zeros((num_epochs,num_trials,num_Fuse))
vld_acc_1Bsg = np.zeros((num_epochs,num_trials,num_Fuse))
trn_acc_1rnd = np.zeros((num_epochs,num_trials,num_Fuse))
vld_acc_1rnd = np.zeros((num_epochs,num_trials,num_Fuse))


#%% Prepare Data

num_classes = 10
# # Loading train set and test set
x_train, x_test, label_train, label_test = get_train_test()
x_train, x_test = standardize(x_train, x_test)


# One-hot encoding:
y_train = to_categorical(label_train)
y_test = to_categorical(label_test)




x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

#z-score
mean = np.mean(x_train,axis=(0,1,2))
std = np.std(x_train,axis=(0,1,2))
x_train = (x_train-mean)/(std+1e-7)
x_test = (x_test-mean)/(std+1e-7)




def lr_schedule(epoch):
    lrate = 0.001
    if epoch > 15:
        lrate = 0.0005
    if epoch > 25:
        lrate = 0.0001
    return lrate

#%% create 2rnd model

for tr in range(0,num_trials):
    L0_2rnd = Input(shape=x_train.shape[1:])
    [Lz,Ld] = Create_rnd_Net(L0_2rnd,x_train.shape[1:],num_classes,n_clayers,weight_decay,True,0,0,0,0,0,0,Sconv,k,n_channels)
    m0 = Model(inputs=L0_2rnd, outputs=Lz)

    model_2rnd = Model(inputs=L0_2rnd, outputs=Ld)
    model_2rnd.summary()
    print('<-Random Initialization2,trial:'+str(tr))
    opt_rms = keras.optimizers.rmsprop(lr=0.001,decay=1e-6)
    model_2rnd.compile(loss='categorical_crossentropy', optimizer=opt_rms, metrics=['accuracy'])
    history_2rnd = model_2rnd.fit(x=x_train, y=y_train,# batch_size=batch_size,\
                        steps_per_epoch=x_train.shape[0] // batch_size,epochs=num_epochs,\
                        validation_steps = x_test.shape[0] // batch_size,\
                        verbose=1,validation_data=(x_test,y_test),callbacks=[LearningRateScheduler(lr_schedule)])
    model_2rnd.save('model_2rnd.h5')



    trn_acc_2rnd[:,tr] = history_2rnd.history['acc']
    vld_acc_2rnd[:,tr] = history_2rnd.history['val_acc']



    W2 = np.transpose(model_2rnd.get_layer('dense').get_weights()[0])
    b2 = model_2rnd.get_layer('dense').get_weights()[1]
    z2_Bsg = (m0.predict(x_train))
    Wc = []
    bc = []
    Wc.append(model_2rnd.get_layer('conv3').get_weights()[0])
    bc.append(model_2rnd.get_layer('conv3').get_weights()[1])
    Wc.append(model_2rnd.get_layer('conv4').get_weights()[0])
    bc.append(model_2rnd.get_layer('conv4').get_weights()[1])
    if(n_clayers>=5):
        Wc.append(model_2rnd.get_layer('conv5').get_weights()[0])
        bc.append(model_2rnd.get_layer('conv5').get_weights()[1])
    if(n_clayers>=6):
        Wc.append(model_2rnd.get_layer('conv6').get_weights()[0])
        bc.append(model_2rnd.get_layer('conv6').get_weights()[1])
    Wd = (model_2rnd.get_layer('dense').get_weights()[0])
    bd = (model_2rnd.get_layer('dense').get_weights()[1])

    for f in range(0,num_Fuse):
        #%% compute Bussgang1 initializer
        epsilon = 1e-6

        #%% compute Bussgang initialization:
        K=1
        n_tr = z2_Bsg.shape[0]
        a0 = x_train[0:n_tr,:,:]
        allN = np.hstack((n_Inchannels,N))
        h_tilde = np.zeros((k_Bsg,allN[K-1], allN[K+1]),dtype=np.float32)
        stilde = Sconv[K-1]*Spool[K-1]*Sconv[K]
        v_p = z2_Bsg - np.mean(z2_Bsg,axis=0)
        L0 = a0.shape[1]
        epsilon = 1e-3
        for mm in range(0,allN[K-1]): # in channel idx of 1st layer
            u_tmp = np.transpose(a0[0:n_tr,:,mm] - np.mean(a0[0:n_tr,:,mm]))
            u = np.vstack((np.zeros((int((k_Bsg-1)/2),n_tr)),np.flipud(u_tmp),np.zeros((int((k_Bsg-1)/2),n_tr))))
            for nn in range(0,allN[K+1]): # output channel idx of 2nd layer N2
                v = np.transpose(v_p[:,:,nn])
    #            v_upsampled = np.insert(vtmp, slice(1, None), 0,axis=0)
    #            v_tilde = np.vstack((np.zeros((int((krnl_sz_Bsg-1)/2),n_tr)),v_upsampled,np.zeros((int((krnl_sz_Bsg-1)/2+1),n_tr))))
                V = np.zeros((k_Bsg,1))
                U = np.zeros((k_Bsg,k_Bsg))
                for ii in range(0,L0,stilde):#range(1,seq_len,2):
                    u_sub = u[L0-ii-1:L0-ii+k_Bsg-1,:]
                    v_sub = v[int(ii/stilde),:]
                    U += np.matmul(u_sub,np.transpose(u_sub))/n_tr
                    V += np.reshape(np.mean(v_sub*u_sub,axis=1),[-1,1]) # check this
                Uinv = np.linalg.inv(U+epsilon*np.eye(k_Bsg,dtype=np.float32))
                h_tilde[:,mm,nn] = np.squeeze(np.matmul(Uinv,V))

        remainNodes_idx = np.array([True,True,True,True,True,True]) # index of those nodes that remain after fusion
        remainNodes_idx[K-1] = False
        rndinitL_1 = remainNodes_idx
        Sconv_Bsg = np.hstack((Sconv[remainNodes_idx],1))
        Sconv_Bsg[K-1] = Sconv[K-1]*Spool[K-1]*Sconv[K]
        N_Bsg = np.hstack((N[remainNodes_idx],1))

        hconvb_tilde = np.zeros((int(L0/stilde),allN[K+1]),dtype=np.float32)
        a0bar = np.mean(a0,axis=0)
        for nnn in range(0,allN[K+1]):
            convtmp = np.zeros((L0),dtype=np.float32)
            for mmm in range(0,allN[K-1]):
                convtmp += np.convolve(h_tilde[:,mmm,nnn],a0bar[:,mmm],mode='same')
            hconvb_tilde[:,nnn] = convtmp[0:L0:stilde]
        b_tilde = np.mean(np.mean(z2_Bsg,axis=0) - hconvb_tilde)


        #% Create Bussgang1 model
        L0_1Bsg = Input(shape=x_train.shape[1:])
        [Lz_Bsg,Ld_Bsg] = Create_rnd_Net(L0_1Bsg,x_train.shape[1:],num_classes,n_clayers-f-1,weight_decay,False
                    ,h_tilde,b_tilde,Wc,bc,Wd,bd,Sconv_Bsg,k_Bsg,N_Bsg)


        m0_1Bsg = Model(inputs=L0_1Bsg, outputs=Lz_Bsg)

        #%training 1Bsg:
        model_1Bsg = Model(inputs=L0_1Bsg, outputs=Ld_Bsg)
        model_1Bsg.summary()
        print('<-BussgangInitialization1,trial:'+str(tr))
        opt_rms = keras.optimizers.rmsprop(lr=0.001,decay=1e-6)
        model_1Bsg.compile(loss='categorical_crossentropy', optimizer=opt_rms, metrics=['accuracy'])
        history_1Bsg = model_1Bsg.fit(x=x_train, y=y_train,# batch_size=batch_size,\
                            steps_per_epoch=x_train.shape[0] // batch_size,epochs=num_epochs,\
                            validation_steps=x_test.shape[0] // batch_size,\
                            verbose=1,validation_data=(x_test,y_test),callbacks=[LearningRateScheduler(lr_schedule)])
        model_1Bsg.save('model_1Bsg.h5')

        trn_acc_1Bsg[:,tr,f] = history_1Bsg.history['acc']
        vld_acc_1Bsg[:,tr,f] = history_1Bsg.history['val_acc']


        if(f < num_Fuse):
            z2_Bsg = np.transpose(m0_1Bsg.predict(x_train))
            Wc = []
            bc = []
            Wc.append(model_1Bsg.get_layer('conv3').get_weights()[0])
            bc.append(model_1Bsg.get_layer('conv3').get_weights()[1])
            if(n_clayers-f-1>=4):
                Wc.append(model_1Bsg.get_layer('conv4').get_weights()[0])
                bc.append(model_1Bsg.get_layer('conv4').get_weights()[1])
            if(n_clayers-f-1>=5):
                Wc.append(model_1Bsg.get_layer('conv5').get_weights()[0])
                bc.append(model_1Bsg.get_layer('conv5').get_weights()[1])
            Wd = (model_1Bsg.get_layer('dense').get_weights()[0])
            bd = (model_1Bsg.get_layer('dense').get_weights()[1])


        #%%training 1rnd
        print('Random Initialization1,trial:'+str(tr)+':')
        L0_1rnd = Input(shape=x_train.shape[1:])
        [tmp,Ld_1rnd] = Create_rnd_Net(L0_1rnd,x_train.shape[1:],num_classes,n_clayers-f-1,weight_decay,True,0,0,0,0,0,0,Sconv_Bsg,k_Bsg,N_Bsg)


        model_1rnd = Model(inputs=L0_1rnd, outputs=Ld_1rnd)
        model_1rnd.summary()
        print('<-Random Initialization1,trial:'+str(tr))
        opt_rms = keras.optimizers.rmsprop(lr=0.001,decay=1e-6)
        model_1rnd.compile(loss='categorical_crossentropy', optimizer=opt_rms, metrics=['accuracy'])
        history_1rnd = model_1rnd.fit(x=x_train,y=y_train,# batch_size=batch_size,\
                            steps_per_epoch=x_train.shape[0] // batch_size,epochs=num_epochs,\
                            validation_steps=x_test.shape[0] // batch_size,\
                            verbose=1,validation_data=(x_test,y_test),callbacks=[LearningRateScheduler(lr_schedule)])
        model_1rnd.save('model_1rnd.h5')

        trn_acc_1rnd[:,tr,f] = history_1rnd.history['acc']
        vld_acc_1rnd[:,tr,f] = history_1rnd.history['val_acc']




#%% save results

savepath = 'results'
filename = ('MultiFuseLastConv_numtrials'+str(num_trials)+'_epochs'+str(num_epochs)+
'_batchsize'+str(batch_size)+'_ncLayers'+str(n_clayers)
+'_Kernel-len'+str(k)
#+'_lrn_rate'+str(lrn_rate)
+'_num-channels'+str(n_channels[0])+'-'+str(n_channels[1])+'-'+str(n_channels[2])+str(n_channels[3])
#+'_rndseed'+str(seed)
+'_he-init'+str(he_init)
+'_initializeALL.pkl')

with open(os.path.join(savepath,filename),'wb') as f:
    pickle.dump([trn_acc_2rnd,vld_acc_2rnd,trn_acc_1rnd,vld_acc_1rnd,trn_acc_1Bsg,vld_acc_1Bsg],f)

#%% Plotting results
Plot_MultiFuseNet(n_clayers,n_clayers,trn_acc_2rnd
                 ,trn_acc_1rnd,trn_acc_1Bsg,vld_acc_2rnd,vld_acc_1rnd,vld_acc_1Bsg,
                  num_epochs)
