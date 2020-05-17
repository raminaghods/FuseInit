# -*- coding: utf-8 -*-
"""
Code for paper:

Ghods, Ramina and Lan, Andrew S and Goldstein, Tom and Studer, Christoph, ``MSE-optimal
neural network initialization via layer fusion", 2020 54th Annual Conference on
Information Sciences and Systems (CISS)

(c) 2020 raminaghods (rghods@cs.cmu.edu)

training and testing HAR dataset using tensorflow

"""
import tensorflow as tf
import numpy as np
from utilities import get_batches
import os


def trntst_HAR(graph,inputs_,labels_,keep_prob_,learning_rate_,cost, optimizer, accuracy,z2,z3,z4,a1,a2,Wd,bd,epochs,X_tr,y_tr,X_vld,y_vld,X_tst,y_tst,batch_size,lrn_rate):

# ### Train the network
    z2_all_out_ch = []
    validation_acc = []
    validation_loss = []

    train_acc = []
    train_loss = []

    with graph.as_default():
        saver = tf.train.Saver()

    with tf.Session(graph=graph) as sess:
        sess.run(tf.global_variables_initializer())

        # Loop over epochs
        for e in range(epochs):
            cntr = 0
            # Loop over batches
            tr_loss = []
            tr_acc = []
            for x,y in get_batches(X_tr, y_tr, batch_size):

                # Feed dictionary
                feed = {inputs_ : x, labels_ : y, keep_prob_ : 0.5, learning_rate_ : lrn_rate}

                # Loss
                loss, _ , acc,Wd_opt,bd_opt = sess.run([cost, optimizer, accuracy,Wd,bd], feed_dict = feed)
                if(e == epochs-1):
                    z2_opt_bt = z2.eval( feed_dict = feed)
                    z3_opt_bt = z3.eval( feed_dict = feed)
                    z4_opt_bt = z4.eval( feed_dict = feed)
                    a1_opt_bt = a1.eval( feed_dict = feed)
                    a2_opt_bt = a2.eval( feed_dict = feed)
#                    gr = tf.get_default_graph()
#                    h1 = gr.get_tensor_by_name('conv1/kernel:0').eval()
#                    b1 = gr.get_tensor_by_name('conv1/bias:0').eval()
#                    h2 = gr.get_tensor_by_name('conv2/kernel:0').eval()
#                    b2 = gr.get_tensor_by_name('conv2/bias:0').eval()
                    if(cntr == 0):
                        z2_opt = z2_opt_bt
                        z3_opt = z3_opt_bt
                        z4_opt = z4_opt_bt
                        a1_opt = a1_opt_bt
                        a2_opt = a2_opt_bt
                    else:
                        z2_opt = np.concatenate((z2_opt,z2_opt_bt),axis=0)
                        z3_opt = np.concatenate((z3_opt,z3_opt_bt),axis=0)
                        z4_opt = np.concatenate((z4_opt,z4_opt_bt),axis=0)
                        a1_opt = np.concatenate((a1_opt,a1_opt_bt),axis=0)
                        a2_opt = np.concatenate((a2_opt,a2_opt_bt),axis=0)
                cntr += 1

#                op = sess.graph.get_operations()
#                for m in op:
#                    print(m.name)
                tr_acc.append(acc)
                tr_loss.append(loss)


            print("Epoch: {}/{}".format(e, epochs),
                          "Train loss: {:6f}".format(np.mean(tr_loss)),
                          "Train acc: {:.6f}".format(np.mean(tr_acc)))


            train_acc.append(np.mean(tr_acc))
            train_loss.append(np.mean(tr_loss))

            val_acc_ = []
            val_loss_ = []

            for x_v, y_v in get_batches(X_vld, y_vld, batch_size):
                # Feed
                feed = {inputs_ : x_v, labels_ : y_v, keep_prob_ : 1.0}

                # Loss
#                        loss_v, acc_v = sess.run([cost, accuracy], feed_dict = feed)
                loss_v = cost.eval(feed_dict = feed)
                acc_v = accuracy.eval(feed_dict = feed)
                val_acc_.append(acc_v)
                val_loss_.append(loss_v)

            # Print info
            print("Epoch: {}/{}".format(e, epochs),
                  "Validation loss: {:6f}".format(np.mean(val_loss_)),
                  "Validation acc: {:.6f}".format(np.mean(val_acc_)))

                    # Store
            validation_acc.append(np.mean(val_acc_))
            validation_loss.append(np.mean(val_loss_))

#        saver_path = os.path.join("checkpoints-cnn","har"+".ckpt")
#        saver.save(sess,saver_path)
        test_acc = []
        for x_t, y_t in get_batches(X_tst, y_tst, batch_size):
            feed = {inputs_: x_t,
                labels_: y_t,
                keep_prob_: 1}

            batch_acc = sess.run(accuracy, feed_dict=feed)
            test_acc.append(batch_acc)
        print("Test accuracy: {:.6f}".format(np.mean(test_acc)))


 # ## Evaluate on test set






    z2_all_out_ch = np.array(z2_all_out_ch)
    train_acc_ar = np.array(train_acc)
    validation_acc_ar = np.array(validation_acc)
    train_loss_ar = np.array(train_loss)
    validation_loss_ar = np.array(validation_loss)
    test_acc_ar = np.mean(test_acc)
    return [train_acc_ar,validation_acc_ar,train_loss_ar,validation_loss_ar,test_acc_ar,z2_opt,z3_opt,z4_opt,a1_opt,a2_opt,Wd_opt,bd_opt]
