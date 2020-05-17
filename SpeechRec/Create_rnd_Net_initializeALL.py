# -*- coding: utf-8 -*-
"""
Code for paper:

Ghods, Ramina and Lan, Andrew S and Goldstein, Tom and Studer, Christoph, ``MSE-optimal
neural network initialization via layer fusion", 2020 54th Annual Conference on
Information Sciences and Systems (CISS)

(c) 2020 raminaghods (rghods@cs.cmu.edu)

"""


from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv1D, MaxPooling1D, Input
from keras import regularizers
import keras.initializers as kinit


def Create_rnd_Net(L0,in_len,num_classes,n_clayers,weight_decay,rnd_init,h_tilde,b_tilde,Wc,bc,wd,bd,S,k=3,n_channels=[64,32,32,32]):
#
    if(rnd_init):
        L1 = Conv1D(n_channels[0], k, padding='same',kernel_initializer=kinit.RandomNormal(), kernel_regularizer=regularizers.l2(weight_decay)
                    ,input_shape=in_len,strides=int(S[0]),name='conv1')(L0)
    else:
        L1 = Conv1D(n_channels[0], k, padding='same',kernel_initializer=kinit.Constant(h_tilde),bias_initializer=kinit.Constant(b_tilde),
                    kernel_regularizer=regularizers.l2(weight_decay),strides=int(S[0]), input_shape=in_len,name='conv1')(L0)
    L2 = Activation('relu')(L1)
    L3 = BatchNormalization()(L2)

    if(n_clayers == 2):
        if(rnd_init):
            L4 = Conv1D(n_channels[1], k, padding='same',kernel_initializer=kinit.RandomNormal(), kernel_regularizer=regularizers.l2(weight_decay),strides=int(S[1]),name='conv2')(L3)
        else:
            L4 = Conv1D(n_channels[1], k, padding='same',kernel_initializer=kinit.Constant(Wc[0]),bias_initializer=kinit.Constant(bc[0])
                    , kernel_regularizer=regularizers.l2(weight_decay),strides=int(S[1]),name='conv2')(L3)
        L5 = Activation('relu')(L4)
        L6 = BatchNormalization()(L5)
        L7 = MaxPooling1D(pool_size=(2))(L6)
        L8 = Dropout(0.2)(L7)
        Lc = Flatten()(L8)
    elif(n_clayers == 3):
        if(rnd_init):
            L4 = Conv1D(n_channels[1], k, padding='same',kernel_initializer=kinit.RandomNormal(), kernel_regularizer=regularizers.l2(weight_decay),strides=int(S[1]),name='conv2')(L3)
        else:
            L4 = Conv1D(n_channels[1], k, padding='same',kernel_initializer=kinit.Constant(Wc[0]),bias_initializer=kinit.Constant(bc[0])
                    , kernel_regularizer=regularizers.l2(weight_decay),strides=int(S[1]),name='conv2')(L3)
        L5 = Activation('relu')(L4)
        L6 = BatchNormalization()(L5)
        L7 = MaxPooling1D(pool_size=(2))(L6)
        L8 = Dropout(0.2)(L7)
        if(rnd_init):
            L9 = Conv1D(n_channels[2], k, padding='same',kernel_initializer=kinit.RandomNormal(), kernel_regularizer=regularizers.l2(weight_decay),strides=int(S[2]),name='conv3')(L8)
        else:
            L9 = Conv1D(n_channels[2], k, padding='same',kernel_initializer=kinit.Constant(Wc[1]),bias_initializer=kinit.Constant(bc[1])
                    , kernel_regularizer=regularizers.l2(weight_decay),strides=int(S[2]),name='conv3')(L8)
        L10 = Activation('relu')(L9)
        L11 = BatchNormalization()(L10)
        Lc = Flatten()(L11)
    elif(n_clayers == 4):
        if(rnd_init):
            L4 = Conv1D(n_channels[1], k, padding='same',kernel_initializer=kinit.RandomNormal(), kernel_regularizer=regularizers.l2(weight_decay),strides=int(S[1]),name='conv2')(L3)
        else:
            L4 = Conv1D(n_channels[1], k, padding='same',kernel_initializer=kinit.Constant(Wc[0]),bias_initializer=kinit.Constant(bc[0])
                    , kernel_regularizer=regularizers.l2(weight_decay),strides=int(S[1]),name='conv2')(L3)
        L5 = Activation('relu')(L4)
        L6 = BatchNormalization()(L5)
        L7 = MaxPooling1D(pool_size=(2))(L6)
        L8 = Dropout(0.2)(L7)
        if(rnd_init):
            L9 = Conv1D(n_channels[2], k, padding='same',kernel_initializer=kinit.RandomNormal(), kernel_regularizer=regularizers.l2(weight_decay),strides=int(S[2]),name='conv3')(L8)
        else:
            L9 = Conv1D(n_channels[2], k, padding='same',kernel_initializer=kinit.Constant(Wc[1]),bias_initializer=kinit.Constant(bc[1])
                    , kernel_regularizer=regularizers.l2(weight_decay),strides=int(S[2]),name='conv3')(L8)
        L10 = Activation('relu')(L9)
        L11 = BatchNormalization()(L10)
        if(rnd_init):
            L12 = Conv1D(n_channels[3], k, padding='same',kernel_initializer=kinit.RandomNormal(), kernel_regularizer=regularizers.l2(weight_decay),strides=int(S[3]),name='conv4')(L11)
        else:
            L12 = Conv1D(n_channels[3], k, padding='same',kernel_initializer=kinit.Constant(Wc[2]),bias_initializer=kinit.Constant(bc[2])
                        , kernel_regularizer=regularizers.l2(weight_decay),strides=int(S[3]),name='conv4')(L11)
        L13 = Activation('relu')(L12)
        L14 = BatchNormalization()(L13)
        L15 = MaxPooling1D(pool_size=(2))(L14)
        L16 = Dropout(0.3)(L15)
        Lc = Flatten()(L16)
    elif(n_clayers == 5):
        if(rnd_init):
            L4 = Conv1D(n_channels[1], k, padding='same',kernel_initializer=kinit.RandomNormal(), kernel_regularizer=regularizers.l2(weight_decay),strides=int(S[1]),name='conv2')(L3)
        else:
            L4 = Conv1D(n_channels[1], k, padding='same',kernel_initializer=kinit.Constant(Wc[0]),bias_initializer=kinit.Constant(bc[0])
                    , kernel_regularizer=regularizers.l2(weight_decay),strides=int(S[1]),name='conv2')(L3)
        L5 = Activation('relu')(L4)
        L6 = BatchNormalization()(L5)
        L7 = MaxPooling1D(pool_size=(2))(L6)
        L8 = Dropout(0.2)(L7)
        if(rnd_init):
            L9 = Conv1D(n_channels[2], k, padding='same',kernel_initializer=kinit.RandomNormal(), kernel_regularizer=regularizers.l2(weight_decay),strides=int(S[2]),name='conv3')(L8)
        else:
            L9 = Conv1D(n_channels[2], k, padding='same',kernel_initializer=kinit.Constant(Wc[1]),bias_initializer=kinit.Constant(bc[1])
                    , kernel_regularizer=regularizers.l2(weight_decay),strides=int(S[2]),name='conv3')(L8)
        L10 = Activation('relu')(L9)
        L11 = BatchNormalization()(L10)
        if(rnd_init):
            L12 = Conv1D(n_channels[3], k, padding='same',kernel_initializer=kinit.RandomNormal(), kernel_regularizer=regularizers.l2(weight_decay),strides=int(S[3]),name='conv4')(L11)
        else:
            L12 = Conv1D(n_channels[3], k, padding='same',kernel_initializer=kinit.Constant(Wc[2]),bias_initializer=kinit.Constant(bc[2])
                        , kernel_regularizer=regularizers.l2(weight_decay),strides=int(S[3]),name='conv4')(L11)
        L13 = Activation('relu')(L12)
        L14 = BatchNormalization()(L13)
        L15 = MaxPooling1D(pool_size=(2))(L14)
        L16 = Dropout(0.3)(L15)
        if(rnd_init):
            L17 = Conv1D(n_channels[4], k, padding='same',kernel_initializer=kinit.RandomNormal(), kernel_regularizer=regularizers.l2(weight_decay),strides=int(S[4]),name='conv5')(L16)
        else:
            L17 = Conv1D(n_channels[4], k, padding='same',kernel_initializer=kinit.Constant(Wc[3]),bias_initializer=kinit.Constant(bc[3])
                        ,kernel_regularizer=regularizers.l2(weight_decay),strides=int(S[4]),name='conv5')(L16)
        L18 = Activation('relu')(L17)
        L19 = BatchNormalization()(L18)
        Lc = Flatten()(L19)
    elif(n_clayers == 6):
        if(rnd_init):
            L4 = Conv1D(n_channels[1], k, padding='same',kernel_initializer=kinit.RandomNormal(), kernel_regularizer=regularizers.l2(weight_decay),strides=int(S[1]),name='conv2')(L3)
        else:
            L4 = Conv1D(n_channels[1], k, padding='same',kernel_initializer=kinit.Constant(Wc[0]),bias_initializer=kinit.Constant(bc[0])
                    , kernel_regularizer=regularizers.l2(weight_decay),strides=int(S[1]),name='conv2')(L3)
        L5 = Activation('relu')(L4)
        L6 = BatchNormalization()(L5)
        L7 = MaxPooling1D(pool_size=(2))(L6)
        L8 = Dropout(0.2)(L7)
        if(rnd_init):
            L9 = Conv1D(n_channels[2], k, padding='same',kernel_initializer=kinit.RandomNormal(), kernel_regularizer=regularizers.l2(weight_decay),strides=int(S[2]),name='conv3')(L8)
        else:
            L9 = Conv1D(n_channels[2], k, padding='same',kernel_initializer=kinit.Constant(Wc[1]),bias_initializer=kinit.Constant(bc[1])
                    , kernel_regularizer=regularizers.l2(weight_decay),strides=int(S[2]),name='conv3')(L8)
        L10 = Activation('relu')(L9)
        L11 = BatchNormalization()(L10)
        if(rnd_init):
            L12 = Conv1D(n_channels[3], k, padding='same',kernel_initializer=kinit.RandomNormal(), kernel_regularizer=regularizers.l2(weight_decay),strides=int(S[3]),name='conv4')(L11)
        else:
            L12 = Conv1D(n_channels[3], k, padding='same',kernel_initializer=kinit.Constant(Wc[2]),bias_initializer=kinit.Constant(bc[2])
                        , kernel_regularizer=regularizers.l2(weight_decay),strides=int(S[3]),name='conv4')(L11)
        L13 = Activation('relu')(L12)
        L14 = BatchNormalization()(L13)
        L15 = MaxPooling1D(pool_size=(2))(L14)
        L16 = Dropout(0.3)(L15)
        if(rnd_init):
            L17 = Conv1D(n_channels[4], k, padding='same',kernel_initializer=kinit.RandomNormal(), kernel_regularizer=regularizers.l2(weight_decay)
                ,strides=int(S[4]),name='conv5')(L16)
        else:
            L17 = Conv1D(n_channels[4], k, padding='same',kernel_initializer=kinit.Constant(Wc[3]),bias_initializer=kinit.Constant(bc[3])
                        ,kernel_regularizer=regularizers.l2(weight_decay),strides=int(S[4]),name='conv5')(L16)
        L18 = Activation('relu')(L17)
        L19 = BatchNormalization()(L18)
        L20 = Conv1D(n_channels[5], k, padding='same',kernel_initializer=kinit.RandomNormal(),
                     kernel_regularizer=regularizers.l2(weight_decay),strides=int(S[5]),name='conv6')(L19)
        L21 = Activation('relu')(L20)
        L22 = BatchNormalization()(L21)
        L23 = MaxPooling1D(pool_size=(2))(L22)
        L24 = Dropout(0.4)(L23)
        Lc = Flatten()(L24)


    if(rnd_init):
        Ld = Dense(num_classes, activation='softmax',name='dense',
                   kernel_initializer=kinit.RandomNormal())(Lc)
    else:
        Ld = Dense(num_classes, activation='softmax',name='dense',
                    kernel_initializer=kinit.Constant(wd),
                    bias_initializer=kinit.Constant(bd))(Lc)
    Lz = (L4)
    return [Lz,Ld]
