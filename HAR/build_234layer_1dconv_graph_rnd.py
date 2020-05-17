# -*- coding: utf-8 -*-
"""
Code for paper:

Ghods, Ramina and Lan, Andrew S and Goldstein, Tom and Studer, Christoph, ``MSE-optimal
neural network initialization via layer fusion", 2020 54th Annual Conference on
Information Sciences and Systems (CISS)

(c) 2020 raminaghods (rghods@cs.cmu.edu)

"""


import tensorflow as tf
import numpy as np
# ### Construct the graph


def build_234layer_1dconv_graph(seq_len, n_channels,n_classes,n_layers,N,krnl_sz,
                                Sconv,Spool,kernel_init=None,b_init=None,rndinit=np.array([True,True,True,True]),rnd_dense=True,Wdo=None,bdo=None):

    tf.reset_default_graph()
    graph = tf.Graph()

    # Construct placeholders
    with graph.as_default():
        inputs_ = tf.placeholder(tf.float32, [None, seq_len, n_channels], name = 'inputs')
        labels_ = tf.placeholder(tf.float32, [None, n_classes], name = 'labels')
        keep_prob_ = tf.placeholder(tf.float32, name = 'keep')
        learning_rate_ = tf.placeholder(tf.float32, name = 'learning_rate')

    length0 = int(seq_len/Sconv[0])
    length1 = int(length0/(Spool[0]*Sconv[1]))
    length2 = int(length1/(Spool[1]*Sconv[2]))
    length3 = int(length2/(Spool[2]*Sconv[3]))
    length4 = int(length3/(Spool[3]))
    if(n_layers==1):
        fcin = length1*Sconv[1]*N[0]
    elif(n_layers==2):
        fcin = length2*Sconv[2]*N[1]
    elif(n_layers==3):
        fcin = length3*Sconv[3]*N[2]
    elif(n_layers==4):
        fcin = length4*N[3]

    #
    # Note: Should we use a different activation? Like tf.nn.tanh?

    # In[9]:


    with graph.as_default():

        if(rndinit[0]):
            b1 = tf.Variable(tf.random_normal([length0,N[0]],stddev=0.01),name='b1')
        else:
            b1 = tf.Variable(b_init,name='b1')

        if(rndinit[1]):
            b2 = tf.Variable(tf.random_normal([length1,N[1]],stddev=0.01),name='b2')
        else:
            b2 = tf.Variable(b_init,name='b2')

        if(rndinit[2]):
            b3 = tf.Variable(tf.random_normal([length2,N[2]],stddev=0.01),name='b3')
        else:
            b3 = tf.Variable(b_init,name='b3')

        if(rndinit[3]):
            b4 = tf.Variable(tf.random_normal([length3,N[3]],stddev=0.01),name='b4')
        else:
            b4 = tf.Variable(b_init,name='b4')

#        if(rnd_dense):
        Wd = tf.Variable(tf.random_normal([n_classes,fcin],stddev=0.05),name='Wd')
        bd = tf.Variable(tf.random_normal([n_classes,1],stddev=0.05),name='bd')
#        else:
#            Wd = tf.Variable(Wdo,name='Wd')
#            bd = tf.Variable(bdo,name='bd')
        #%% # Build Convolutional Layers
        #input dimensions definition: (batchsize,length of sequence,number of input channels)
        #output dimensions definition:  (batchsize,length of sequence,number of output channels)
        if(rndinit[0]):
            # (batch, 128, 9) --> (batch, 64, 18)
            conv1_Nobias = tf.layers.conv1d(inputs=inputs_, filters=N[0], kernel_size=krnl_sz, strides=int(Sconv[0]),padding='same',
                                    kernel_initializer=tf.initializers.random_normal(stddev=0.05),
                                     use_bias=False,name='conv1_Nobias')# filter: (krnl_sz,n_channels,N[0])), conv1: (batch,seq_len,N[0]])


        else:
                        # (batch, 128, 9) --> (batch, 64, 18)
            conv1_Nobias = tf.layers.conv1d(inputs=inputs_, filters=N[0], kernel_size=krnl_sz, strides=int(Sconv[0]),padding='same',
                                     kernel_initializer=tf.constant_initializer(value=kernel_init),
                                     use_bias=False, name='conv1_Nobias')# filter: (krnl_sz,n_channels,N[0])), conv1: (batch,seq_len,N[0]])

        conv1 = tf.add(conv1_Nobias,b1)
        conv1_relu = tf.nn.relu(conv1)
        max_pool_1 = tf.layers.max_pooling1d(inputs=conv1_relu, pool_size=int(Spool[0]), strides=int(Spool[0]), padding='same')

        if(rndinit[1]):
            # (batch, 64, 18) --> (batch, 32, 36)
            conv2_Nobias = tf.layers.conv1d(inputs=max_pool_1, filters=N[1], kernel_size=krnl_sz, strides=int(Sconv[1]),
                                            kernel_initializer=tf.initializers.random_normal(stddev=0.05),
                                     use_bias=False,padding='same', name='conv2_Nobias')
        else:
            conv2_Nobias = tf.layers.conv1d(inputs=max_pool_1, filters=N[1], kernel_size=krnl_sz, strides=int(Sconv[1]),padding='same',
                                     kernel_initializer=tf.constant_initializer(value=kernel_init),
                                     use_bias=False, name='conv2_Nobias')
        conv2 = tf.add(conv2_Nobias,b2)
        conv2_relu = tf.nn.relu(conv2)
        max_pool_2 = tf.layers.max_pooling1d(inputs=conv2_relu, pool_size=int(Spool[1]), strides=int(Spool[1]), padding='same')

        if(rndinit[2]):
            # (batch, 32, 36) --> (batch, 16, 72)
            conv3_Nobias = tf.layers.conv1d(inputs=max_pool_2, filters=N[2], kernel_size=krnl_sz, strides=int(Sconv[2]),
                                    kernel_initializer=tf.initializers.random_normal(stddev=0.05),
                                 use_bias=False,padding='same', name='conv3_Nobias')
        else:
            conv3_Nobias = tf.layers.conv1d(inputs=max_pool_2, filters=N[2], kernel_size=krnl_sz, strides=int(Sconv[2]),
                                 kernel_initializer=tf.constant_initializer(value=kernel_init),
                                 use_bias=False,padding='same', name='conv3_Nobias')

        conv3 = tf.add(conv3_Nobias,b3)
        conv3_relu = tf.nn.relu(conv3)
        max_pool_3 = tf.layers.max_pooling1d(inputs=conv3_relu, pool_size=int(Spool[2]), strides=int(Spool[2]), padding='same')

        if(rndinit[3]):
            # (batch, 16, 72) --> (batch, 8, 144)
            conv4_Nobias = tf.layers.conv1d(inputs=max_pool_3, filters=N[3], kernel_size=krnl_sz, strides=int(Sconv[3]),
                                    kernel_initializer=tf.initializers.random_normal(stddev=0.05),
                                 use_bias=False,padding='same',name='conv4_Nobias')
        else:
            conv4_Nobias = tf.layers.conv1d(inputs=max_pool_3, filters=N[3], kernel_size=krnl_sz, strides=int(Sconv[3]),
                                   kernel_initializer=tf.constant_initializer(value=kernel_init),
                                   use_bias=False,padding='same',name='conv4_Nobias')
        conv4 = tf.add(conv4_Nobias,b4)
        conv4_relu = tf.nn.relu(conv4)
        max_pool_4 = tf.layers.max_pooling1d(inputs=conv4_relu, pool_size=int(Spool[3]), strides=int(Spool[3]), padding='same')


    # Now, flatten and pass to the classifier

    # In[10]:


    with graph.as_default():
        # Flatten and add dropout
        if(n_layers==1):
            flat = tf.reshape(max_pool_1, (-1,max_pool_1.shape[1]*max_pool_1.shape[2]))
        elif(n_layers==2):
            flat = tf.reshape(max_pool_2, (-1,max_pool_2.shape[1]*max_pool_2.shape[2]))
        elif(n_layers==3):
            flat = tf.reshape(max_pool_3, (-1,max_pool_3.shape[1]*max_pool_3.shape[2]))
        elif(n_layers==4):
            flat = tf.reshape(max_pool_4, (-1,max_pool_4.shape[1]*max_pool_4.shape[2]))
        else:
            print('wrong number of layers')

        flat = tf.nn.dropout(flat, keep_prob=keep_prob_)

        # Predictions
        out = tf.add(tf.matmul(Wd, tf.transpose(flat)), bd)
        logits = tf.nn.softmax(tf.transpose(out))
#        logits = tf.layers.dense(flat, n_classes)

        # Cost function and optimizer
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels_))
        optimizer = tf.train.AdamOptimizer(learning_rate_).minimize(cost)

        # Accuracy
        correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(labels_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')



    return [graph,inputs_,labels_,keep_prob_,learning_rate_,cost, optimizer, accuracy,conv2,conv3,conv4,max_pool_1,max_pool_2,Wd,bd]
