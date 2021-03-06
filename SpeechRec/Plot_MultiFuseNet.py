# -*- coding: utf-8 -*-
"""
Code for paper:

Ghods, Ramina and Lan, Andrew S and Goldstein, Tom and Studer, Christoph, ``MSE-optimal
neural network initialization via layer fusion", 2020 54th Annual Conference on
Information Sciences and Systems (CISS)

(c) 2020 raminaghods (rghods@cs.cmu.edu)

"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from matplotlib2tikz import save as tikz_save


def Plot_MultiFuseNet(L,fuse_layer_idx,train_acc_2rnd
                 ,train_acc_1rnd,train_acc_1Bsg,validation_acc_2rnd,validation_acc_1rnd,validation_acc_1Bsg
                 ,epochs):

    # Plot training and test loss
    plt.close('all')

    colors = ['g','r','y','c','m','k','orange','pink','brown','purple','turquoise']
    t = np.arange(epochs)
    num_fuse = validation_acc_1Bsg.shape[2]

    plt.figure(figsize = (8,6))
    sb.tsplot(time=t,data=np.transpose(train_acc_2rnd), ci='sd', color='b')
    for f in range(0,num_fuse):
        sb.tsplot(time=t,data=np.transpose(train_acc_1rnd[:,:,f]), ci='sd', color=colors[2*f])
        sb.tsplot(time=t,data=np.transpose(train_acc_1Bsg[:,:,f]), ci='sd', color=colors[2*f+1])
    plt.xlabel("iteration")
    plt.ylabel("training Accuracy")
    plt.title('Fused layer: %1.0f'%fuse_layer_idx)
    plt.legend(['%3.0f layer Random'%L , '%3.0f layer Random'%(L-1),'%3.0f layer FuseNet'%(L-1)
                            , '%3.0f layer Random'%(L-2),'%3.0f layer FuseNet'%(L-2)
                         , '%3.0f layer Random'%(L-3),'%3.0f layer FuseNet'%(L-3)
                         , '%3.0f layer Random'%(L-4),'%3.0f layer FuseNet'%(L-4)
                         , '%3.0f layer Random'%(L-5),'%3.0f layer FuseNet'%(L-5)], loc='upper left')
    plt.show()

    plt.figure(figsize = (8,6))
    sb.tsplot(time=t,data=np.transpose(validation_acc_2rnd), ci='sd', color='b')
    for f in range(0,num_fuse):
        sb.tsplot(time=t,data=np.transpose(validation_acc_1rnd[:,:,f]), ci='sd', color=colors[2*f])
        sb.tsplot(time=t,data=np.transpose(validation_acc_1Bsg[:,:,f]), ci='sd', color=colors[2*f+1])
    plt.xlabel("iteration",fontsize = 18)
    plt.ylabel("validation Accuracy",fontsize = 18)
    plt.xticks(fontsize = 18)
    plt.yticks(fontsize = 18)
    plt.legend(['%3.0f layer Random'%L , '%3.0f layer Random'%(L-1),'%3.0f layer FuseNet'%(L-1)
                            , '%3.0f layer Random'%(L-2),'%3.0f layer FuseNet'%(L-2)
                         , '%3.0f layer Random'%(L-3),'%3.0f layer FuseNet'%(L-3)
                         , '%3.0f layer Random'%(L-4),'%3.0f layer FuseNet'%(L-4)
                         , '%3.0f layer Random'%(L-5),'%3.0f layer FuseNet'%(L-5)], loc='upper left')
#    plt.title('Fused layer: %1.0f'%fuse_layer_idx)
#    plt.legend(['%3.0f layer Random'%L , '%3.0f layer Random'%(L-1),'%3.0f layer FuseNet'%(L-1)], loc='center right',fontsize = 16)
#    s = "%3.0f layer FuseNet - test accuracy: %3.3f" %(L-1,np.mean(test_acc_1Bsg))
#    plt. text(600,0.1, s,fontsize = 16, bbox=dict(facecolor='red', alpha=0.2))
#    s = "%3.0f layer Random - test accuracy: %3.3f" %(L-1,np.mean(test_acc_1rnd))
#    plt. text(600,0.17, s,fontsize = 16, bbox=dict(facecolor='green', alpha=0.2))
#    s = "%3.0f layer Random - test accuracy: %3.3f" %(L,np.mean(test_acc_2rnd))
#    plt. text(600,0.24, s,fontsize = 16, bbox=dict(facecolor='blue', alpha=0.2))
    plt.show()
    tikz_save('TexFile.tex')
