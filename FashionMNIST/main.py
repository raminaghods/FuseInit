# -*- coding: utf-8 -*-
"""
Code for paper:

Ghods, Ramina and Lan, Andrew S and Goldstein, Tom and Studer, Christoph, ``MSE-optimal
neural network initialization via layer fusion", 2020 54th Annual Conference on
Information Sciences and Systems (CISS)

(c) 2020 raminaghods (rghods@cs.cmu.edu)

fuses multiple convolutional layers from last layer of the network one at a time
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
from Create_rnd_Net import Create_rnd_Net
from Plot_MultiFuseNet import Plot_MultiFuseNet


#%% parameters
n_clayers = 4
num_Fuse = 3
num_trials = 10
num_epochs = 50
weight_decay = 1e-4
batch_size = 64
k = 3
n_channels = [2,4,8,16]

#%%initialize
trn_acc_2rnd = np.zeros((num_epochs,num_trials))
vld_acc_2rnd = np.zeros((num_epochs,num_trials))
trn_acc_1Bsg = np.zeros((num_epochs,num_trials,num_Fuse))
vld_acc_1Bsg = np.zeros((num_epochs,num_trials,num_Fuse))
trn_acc_1rnd = np.zeros((num_epochs,num_trials,num_Fuse))
vld_acc_1rnd = np.zeros((num_epochs,num_trials,num_Fuse))




def lr_schedule(epoch):
    lrate = 0.001
    if epoch > 15:
        lrate = 0.0005
    if epoch > 25:
        lrate = 0.0003
    return lrate

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train = np.expand_dims(x_train,axis=3)
x_test = np.expand_dims(x_test,axis=3)

#z-score
mean = np.mean(x_train,axis=(0,1,2,3))
std = np.std(x_train,axis=(0,1,2,3))
x_train = (x_train-mean)/(std+1e-7)
x_test = (x_test-mean)/(std+1e-7)

num_classes = 10
y_train = np_utils.to_categorical(y_train,num_classes)
y_test = np_utils.to_categorical(y_test,num_classes)

#data augmentation
datagen = ImageDataGenerator(
#    rotation_range=15,
#    width_shift_range=0.1,
#    height_shift_range=0.1,
#    horizontal_flip=True,
    )
datagen.fit(x_train)


#%% create 2rnd model

for tr in range(0,num_trials):
    L0 = Input(shape=x_train.shape[1:])
    [Lc_a0,Lc_a1,Ld] = Create_rnd_Net(L0,x_train.shape[1:],num_classes,n_clayers,weight_decay,True,0,0,k,n_channels)
    m0 = Model(inputs=L0, outputs=Lc_a0)
    m1 = Model(inputs=L0, outputs=Lc_a1)

    model_2rnd = Model(inputs=L0, outputs=Ld)
    model_2rnd.summary()
    print('<-Random Initialization2,trial:'+str(tr))
    opt_rms = keras.optimizers.rmsprop(lr=0.001,decay=1e-6)
    model_2rnd.compile(loss='categorical_crossentropy', optimizer=opt_rms, metrics=['accuracy'])
    history_2rnd = model_2rnd.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),\
                        steps_per_epoch=x_train.shape[0] // batch_size,epochs=num_epochs,\
                        verbose=1,validation_data=(x_test,y_test),callbacks=[LearningRateScheduler(lr_schedule)])
    model_2rnd.save('model_2rnd.h5')



    trn_acc_2rnd[:,tr] = history_2rnd.history['acc']
    vld_acc_2rnd[:,tr] = history_2rnd.history['val_acc']



    W2 = np.transpose(model_2rnd.get_layer('dense').get_weights()[0])
    b2 = model_2rnd.get_layer('dense').get_weights()[1]
    a0 = np.transpose(m0.predict(x_train))
    a1 = np.transpose(m1.predict(x_train))


    for f in range(0,num_Fuse):
        #%% compute Bussgang1 initializer
        epsilon = 1e-6
        num_trn = a0.shape[1]
        a1_bar = np.reshape(np.mean(a1,axis=1),[-1,1])
        a0_bar = np.reshape(np.mean(a0,axis=1),[-1,1])
        a1_nomean = a1 - a1_bar
        a0_nomean = a0 - a0_bar

        C_a0 = np.matmul(a0_nomean,np.transpose(np.conj(a0_nomean)))/num_trn
        C_W2a1a0 = np.matmul(np.matmul(W2,a1_nomean),np.transpose(np.conj(a0_nomean)))/num_trn

        print('BussgangInitialization1,trial:'+str(tr)+':')
        w_init = np.matmul(C_W2a1a0,np.linalg.inv(C_a0+epsilon*np.eye(a0.shape[0],dtype=np.float32)))
        b_init = np.matmul(W2,a1_bar) - np.matmul(w_init,a0_bar) + np.reshape(b2,[-1,1])


        #% Create Bussgang1 model
        L0_1Bsg = Input(shape=x_train.shape[1:])
        [Lc_a0_Bsg,Lc_a1_Bsg,Ld_Bsg] = Create_rnd_Net(L0_1Bsg,x_train.shape[1:],num_classes,n_clayers-f-1,weight_decay,False
                    ,np.transpose(w_init),np.squeeze(b_init),k,n_channels)


        m0_1Bsg = Model(inputs=L0_1Bsg, outputs=Lc_a0_Bsg)
        m1_1Bsg = Model(inputs=L0_1Bsg, outputs=Lc_a1_Bsg)

        #%training 1Bsg:
        model_1Bsg = Model(inputs=L0_1Bsg, outputs=Ld_Bsg)
        model_1Bsg.summary()
        print('<-BussgangInitialization1,trial:'+str(tr))
        opt_rms = keras.optimizers.rmsprop(lr=0.001,decay=1e-6)
        model_1Bsg.compile(loss='categorical_crossentropy', optimizer=opt_rms, metrics=['accuracy'])
        history_1Bsg = model_1Bsg.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),\
                            steps_per_epoch=x_train.shape[0] // batch_size,epochs=num_epochs,\
                            verbose=1,validation_data=(x_test,y_test),callbacks=[LearningRateScheduler(lr_schedule)])
        model_1Bsg.save('model_1Bsg.h5')

        trn_acc_1Bsg[:,tr,f] = history_1Bsg.history['acc']
        vld_acc_1Bsg[:,tr,f] = history_1Bsg.history['val_acc']


        if(f < num_Fuse):
            W2 = np.transpose(model_1Bsg.get_layer('dense').get_weights()[0])
            b2 = model_1Bsg.get_layer('dense').get_weights()[1]
            a0 = np.transpose(m0_1Bsg.predict(x_train))
            a1 = np.transpose(m1_1Bsg.predict(x_train))


        #%%training 1rnd
        print('Random Initialization1,trial:'+str(tr)+':')
        L0_1rnd = Input(shape=x_train.shape[1:])
        [tmp,tmp,Ld_1rnd] = Create_rnd_Net(L0_1rnd,x_train.shape[1:],num_classes,n_clayers-f-1,weight_decay,True,0,0,k,n_channels)


        model_1rnd = Model(inputs=L0_1rnd, outputs=Ld_1rnd)
        model_1rnd.summary()
        print('<-Random Initialization1,trial:'+str(tr))
        opt_rms = keras.optimizers.rmsprop(lr=0.001,decay=1e-6)
        model_1rnd.compile(loss='categorical_crossentropy', optimizer=opt_rms, metrics=['accuracy'])
        history_1rnd = model_1rnd.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),\
                            steps_per_epoch=x_train.shape[0] // batch_size,epochs=num_epochs,\
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
+'.pkl')

with open(os.path.join(savepath,filename),'wb') as f:
    pickle.dump([trn_acc_2rnd,vld_acc_2rnd,trn_acc_1rnd,vld_acc_1rnd,trn_acc_1Bsg,vld_acc_1Bsg],f)

#%% Plotting results
Plot_MultiFuseNet(n_clayers,n_clayers,trn_acc_2rnd
                 ,trn_acc_1rnd,trn_acc_1Bsg,vld_acc_2rnd,vld_acc_1rnd,vld_acc_1Bsg,
                  num_epochs)
