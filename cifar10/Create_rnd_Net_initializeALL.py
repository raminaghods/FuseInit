# -*- coding: utf-8 -*-
"""
Code for paper:

Ghods, Ramina and Lan, Andrew S and Goldstein, Tom and Studer, Christoph, ``MSE-optimal
neural network initialization via layer fusion", 2020 54th Annual Conference on
Information Sciences and Systems (CISS)

(c) 2020 raminaghods (rghods@cs.cmu.edu)

"""


from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Input
from keras import regularizers
import keras.initializers as kinit


def Create_rnd_Net(L0,in_len,num_classes,n_clayers,weight_decay,rnd_init,w_init,b_init,Wc,bc):
#
    k = 3
    if(rnd_init):
        L1 = Conv2D(32, (3,3), padding='same',kernel_initializer=kinit.RandomNormal(), kernel_regularizer=regularizers.l2(weight_decay)
                    ,input_shape=in_len,name='conv1')(L0)
    else:
        L1 = Conv2D(32, (3,3), padding='same',kernel_initializer=kinit.Constant(Wc[0]),bias_initializer=kinit.Constant(bc[0]),
                    kernel_regularizer=regularizers.l2(weight_decay), input_shape=in_len,name='conv1')(L0)
    L2 = Activation('elu')(L1)
    L3 = BatchNormalization()(L2)
    if(n_clayers == 1):
        Lc_a0 = Flatten()(L0)
        Lc_a1 = Flatten()(L3)
    elif(n_clayers == 2):
        Lc_a0 = Flatten()(L3)
        if(rnd_init):
            L4 = Conv2D(32, (k,k), padding='same',kernel_initializer=kinit.RandomNormal(), kernel_regularizer=regularizers.l2(weight_decay),name='conv2')(L3)
        else:
            L4 = Conv2D(32, (k,k), padding='same',kernel_initializer=kinit.Constant(Wc[1]),bias_initializer=kinit.Constant(bc[1])
                    , kernel_regularizer=regularizers.l2(weight_decay),name='conv2')(L3)
        L5 = Activation('elu')(L4)
        L6 = BatchNormalization()(L5)
        L7 = MaxPooling2D(pool_size=(2,2))(L6)
        L8 = Dropout(0.2)(L7)
        Lc_a1 = Flatten()(L8)
    elif(n_clayers == 3):
        if(rnd_init):
            L4 = Conv2D(32, (k,k), padding='same',kernel_initializer=kinit.RandomNormal(), kernel_regularizer=regularizers.l2(weight_decay),name='conv2')(L3)
        else:
            L4 = Conv2D(32, (k,k), padding='same',kernel_initializer=kinit.Constant(Wc[1]),bias_initializer=kinit.Constant(bc[1])
                    , kernel_regularizer=regularizers.l2(weight_decay),name='conv2')(L3)
        L5 = Activation('elu')(L4)
        L6 = BatchNormalization()(L5)
        L7 = MaxPooling2D(pool_size=(2,2))(L6)
        L8 = Dropout(0.2)(L7)
        if(rnd_init):
            L9 = Conv2D(64, (k,k), padding='same',kernel_initializer=kinit.RandomNormal(), kernel_regularizer=regularizers.l2(weight_decay),name='conv3')(L8)
        else:
            L9 = Conv2D(64, (k,k), padding='same',kernel_initializer=kinit.Constant(Wc[2]),bias_initializer=kinit.Constant(bc[2])
                    , kernel_regularizer=regularizers.l2(weight_decay),name='conv3')(L8)
        L10 = Activation('elu')(L9)
        L11 = BatchNormalization()(L10)
        Lc_a1 = Flatten()(L11)
        Lc_a0 = Flatten()(L8)
    elif(n_clayers == 4):
        if(rnd_init):
            L4 = Conv2D(32, (k,k), padding='same',kernel_initializer=kinit.RandomNormal(), kernel_regularizer=regularizers.l2(weight_decay),name='conv2')(L3)
        else:
            L4 = Conv2D(32, (k,k), padding='same',kernel_initializer=kinit.Constant(Wc[1]),bias_initializer=kinit.Constant(bc[1])
                    , kernel_regularizer=regularizers.l2(weight_decay),name='conv2')(L3)
        L5 = Activation('elu')(L4)
        L6 = BatchNormalization()(L5)
        L7 = MaxPooling2D(pool_size=(2,2))(L6)
        L8 = Dropout(0.2)(L7)
        if(rnd_init):
            L9 = Conv2D(64, (k,k), padding='same',kernel_initializer=kinit.RandomNormal(), kernel_regularizer=regularizers.l2(weight_decay),name='conv3')(L8)
        else:
            L9 = Conv2D(64, (k,k), padding='same',kernel_initializer=kinit.Constant(Wc[2]),bias_initializer=kinit.Constant(bc[2])
                    , kernel_regularizer=regularizers.l2(weight_decay),name='conv3')(L8)
        L10 = Activation('elu')(L9)
        L11 = BatchNormalization()(L10)
        if(rnd_init):
            L12 = Conv2D(64, (k,k), padding='same',kernel_initializer=kinit.RandomNormal(), kernel_regularizer=regularizers.l2(weight_decay),name='conv4')(L11)
        else:
            L12 = Conv2D(64, (k,k), padding='same',kernel_initializer=kinit.Constant(Wc[3]),bias_initializer=kinit.Constant(bc[3])
                        , kernel_regularizer=regularizers.l2(weight_decay),name='conv4')(L11)
        L13 = Activation('elu')(L12)
        L14 = BatchNormalization()(L13)
        L15 = MaxPooling2D(pool_size=(2,2))(L14)
        L16 = Dropout(0.3)(L15)
        Lc_a1 = Flatten()(L16)
        Lc_a0 = Flatten()(L11)
    elif(n_clayers == 5):
        if(rnd_init):
            L4 = Conv2D(32, (k,k), padding='same',kernel_initializer=kinit.RandomNormal(), kernel_regularizer=regularizers.l2(weight_decay),name='conv2')(L3)
        else:
            L4 = Conv2D(32, (k,k), padding='same',kernel_initializer=kinit.Constant(Wc[1]),bias_initializer=kinit.Constant(bc[1])
                    , kernel_regularizer=regularizers.l2(weight_decay),name='conv2')(L3)
        L5 = Activation('elu')(L4)
        L6 = BatchNormalization()(L5)
        L7 = MaxPooling2D(pool_size=(2,2))(L6)
        L8 = Dropout(0.2)(L7)
        if(rnd_init):
            L9 = Conv2D(64, (k,k), padding='same',kernel_initializer=kinit.RandomNormal(), kernel_regularizer=regularizers.l2(weight_decay),name='conv3')(L8)
        else:
            L9 = Conv2D(64, (k,k), padding='same',kernel_initializer=kinit.Constant(Wc[2]),bias_initializer=kinit.Constant(bc[2])
                    , kernel_regularizer=regularizers.l2(weight_decay),name='conv3')(L8)
        L10 = Activation('elu')(L9)
        L11 = BatchNormalization()(L10)
        if(rnd_init):
            L12 = Conv2D(64, (k,k), padding='same',kernel_initializer=kinit.RandomNormal(), kernel_regularizer=regularizers.l2(weight_decay),name='conv4')(L11)
        else:
            L12 = Conv2D(64, (k,k), padding='same',kernel_initializer=kinit.Constant(Wc[3]),bias_initializer=kinit.Constant(bc[3])
                        , kernel_regularizer=regularizers.l2(weight_decay),name='conv4')(L11)
        L13 = Activation('elu')(L12)
        L14 = BatchNormalization()(L13)
        L15 = MaxPooling2D(pool_size=(2,2))(L14)
        L16 = Dropout(0.3)(L15)
        if(rnd_init):
            L17 = Conv2D(128, (k,k), padding='same',kernel_initializer=kinit.RandomNormal(), kernel_regularizer=regularizers.l2(weight_decay),name='conv5')(L16)
        else:
            L17 = Conv2D(128, (k,k), padding='same',kernel_initializer=kinit.Constant(Wc[4]),bias_initializer=kinit.Constant(bc[4])
                        ,kernel_regularizer=regularizers.l2(weight_decay),name='conv5')(L16)
        L18 = Activation('elu')(L17)
        L19 = BatchNormalization()(L18)
        Lc_a1 = Flatten()(L19)
        Lc_a0 = Flatten()(L16)
    elif(n_clayers == 6):
        if(rnd_init):
            L4 = Conv2D(32, (k,k), padding='same',kernel_initializer=kinit.RandomNormal(), kernel_regularizer=regularizers.l2(weight_decay),name='conv2')(L3)
        else:
            L4 = Conv2D(32, (k,k), padding='same',kernel_initializer=kinit.Constant(Wc[1]),bias_initializer=kinit.Constant(bc[1])
                    , kernel_regularizer=regularizers.l2(weight_decay),name='conv2')(L3)
        L5 = Activation('elu')(L4)
        L6 = BatchNormalization()(L5)
        L7 = MaxPooling2D(pool_size=(2,2))(L6)
        L8 = Dropout(0.2)(L7)
        if(rnd_init):
            L9 = Conv2D(64, (k,k), padding='same',kernel_initializer=kinit.RandomNormal(), kernel_regularizer=regularizers.l2(weight_decay),name='conv3')(L8)
        else:
            L9 = Conv2D(64, (k,k), padding='same',kernel_initializer=kinit.Constant(Wc[2]),bias_initializer=kinit.Constant(bc[2])
                    , kernel_regularizer=regularizers.l2(weight_decay),name='conv3')(L8)
        L10 = Activation('elu')(L9)
        L11 = BatchNormalization()(L10)
        if(rnd_init):
            L12 = Conv2D(64, (k,k), padding='same',kernel_initializer=kinit.RandomNormal(), kernel_regularizer=regularizers.l2(weight_decay),name='conv4')(L11)
        else:
            L12 = Conv2D(64, (k,k), padding='same',kernel_initializer=kinit.Constant(Wc[3]),bias_initializer=kinit.Constant(bc[3])
                        , kernel_regularizer=regularizers.l2(weight_decay),name='conv4')(L11)
        L13 = Activation('elu')(L12)
        L14 = BatchNormalization()(L13)
        L15 = MaxPooling2D(pool_size=(2,2))(L14)
        L16 = Dropout(0.3)(L15)
        if(rnd_init):
            L17 = Conv2D(128, (k,k), padding='same',kernel_initializer=kinit.RandomNormal(), kernel_regularizer=regularizers.l2(weight_decay),name='conv5')(L16)
        else:
            L17 = Conv2D(128, (k,k), padding='same',kernel_initializer=kinit.Constant(Wc[4]),bias_initializer=kinit.Constant(bc[4])
                        ,kernel_regularizer=regularizers.l2(weight_decay),name='conv5')(L16)
        L18 = Activation('elu')(L17)
        L19 = BatchNormalization()(L18)
        L20 = Conv2D(128, (k,k), padding='same',kernel_initializer=kinit.RandomNormal(), kernel_regularizer=regularizers.l2(weight_decay),name='conv6')(L19)
        L21 = Activation('elu')(L20)
        L22 = BatchNormalization()(L21)
        L23 = MaxPooling2D(pool_size=(2,2))(L22)
        L24 = Dropout(0.4)(L23)
        Lc_a1 = Flatten()(L24)
        Lc_a0 = Flatten()(L19)

    if(rnd_init):
        Ld = Dense(num_classes, activation='softmax',name='dense',
                   kernel_initializer=kinit.RandomNormal())(Lc_a1)
    else:
        Ld = Dense(num_classes, activation='softmax',name='dense',
                    kernel_initializer=kinit.Constant(w_init),
                    bias_initializer=kinit.Constant(b_init))(Lc_a1)

    return [Lc_a0,Lc_a1,Ld]
