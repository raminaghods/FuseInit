# -*- coding: utf-8 -*-
"""
Code for paper:

Ghods, Ramina and Lan, Andrew S and Goldstein, Tom and Studer, Christoph, ``MSE-optimal
neural network initialization via layer fusion", 2020 54th Annual Conference on
Information Sciences and Systems (CISS)

(c) 2020 raminaghods (rghods@cs.cmu.edu)

training and testing HAR dataset using tensorflow

"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from matplotlib2tikz import save as tikz_save



def Plot_FuseNet(L,fuse_layer_idx,train_loss_2rnd,train_loss_1rnd,train_loss_1Bsg,validation_loss_2rnd,validation_loss_1rnd,validation_loss_1Bsg,train_acc_2rnd
                 ,train_acc_1rnd,train_acc_1Bsg,validation_acc_2rnd,validation_acc_1rnd,validation_acc_1Bsg,test_acc_2rnd,test_acc_1rnd,test_acc_1Bsg,
                  epochs):

    # Plot training and test loss
    plt.close('all')

    t = np.arange(epochs)

    plt.figure(figsize = (8,6))
    sb.tsplot(time=t,data=np.transpose(train_loss_2rnd), ci='sd', color='b')
    sb.tsplot(time=t,data=np.transpose(train_loss_1rnd), ci='sd', color='g')
    sb.tsplot(time=t,data=np.transpose(train_loss_1Bsg), ci='sd', color='r')
    plt.xlabel("iteration")
    plt.ylabel("training Loss")
    plt.title('Fused layer: %1.0f'%fuse_layer_idx)
    plt.legend(['%3.0f layer Random'%L , '%3.0f layer Random'%(L-1),'%3.0f layer FuseNet'%(L-1)], loc='upper left')
    plt.show()

    plt.figure(figsize = (8,6))
    sb.tsplot(time=t,data=np.transpose(validation_loss_2rnd), ci='sd', color='b')
    sb.tsplot(time=t,data=np.transpose(validation_loss_1rnd), ci='sd', color='g')
    sb.tsplot(time=t,data=np.transpose(validation_loss_1Bsg), ci='sd', color='r')
    plt.xlabel("iteration")
    plt.ylabel("validation Loss")
    plt.title('Fused layer: %1.0f'%fuse_layer_idx)
    plt.legend(['%3.0f layer Random'%L , '%3.0f layer Random'%(L-1),'%3.0f layer FuseNet'%(L-1)], loc='upper left')
    plt.show()

    plt.figure(figsize = (8,6))
    sb.tsplot(time=t,data=np.transpose(train_acc_2rnd), ci='sd', color='b')
    sb.tsplot(time=t,data=np.transpose(train_acc_1rnd), ci='sd', color='g')
    sb.tsplot(time=t,data=np.transpose(train_acc_1Bsg), ci='sd', color='r')
    plt.xlabel("iteration")
    plt.ylabel("training Accuracy")
    plt.title('Fused layer: %1.0f'%fuse_layer_idx)
    plt.legend(['%3.0f layer Random'%L , '%3.0f layer Random'%(L-1),'%3.0f layer FuseNet'%(L-1)], loc='upper left')
    plt.show()

    plt.figure(figsize = (8,6))
    sb.tsplot(time=t,data=np.transpose(validation_acc_2rnd), ci='sd', color='b')
    sb.tsplot(time=t,data=np.transpose(validation_acc_1rnd), ci='sd', color='g')
    sb.tsplot(time=t,data=np.transpose(validation_acc_1Bsg), ci='sd', color='r')
    plt.xlabel("iteration",fontsize = 18)
    plt.ylabel("validation Accuracy",fontsize = 18)
    plt.xticks(fontsize = 18)
    plt.yticks(fontsize = 18)
#    plt.title('Fused layer: %1.0f'%fuse_layer_idx)
#    plt.legend(['%3.0f layer Random'%L , '%3.0f layer Random'%(L-1),'%3.0f layer FuseNet'%(L-1)], loc='center right',fontsize = 16)
    s = "%3.0f layer FuseNet - test accuracy: %3.3f" %(L-1,np.mean(test_acc_1Bsg))
    plt. text(600,0.1, s,fontsize = 16, bbox=dict(facecolor='red', alpha=0.2))
    s = "%3.0f layer Random - test accuracy: %3.3f" %(L-1,np.mean(test_acc_1rnd))
    plt. text(600,0.17, s,fontsize = 16, bbox=dict(facecolor='green', alpha=0.2))
    s = "%3.0f layer Random - test accuracy: %3.3f" %(L,np.mean(test_acc_2rnd))
    plt. text(600,0.24, s,fontsize = 16, bbox=dict(facecolor='blue', alpha=0.2))
    plt.show()
    tikz_save('TexFile.tex')
