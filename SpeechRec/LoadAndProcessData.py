# -*- coding: utf-8 -*-
"""
Code for paper:

Ghods, Ramina and Lan, Andrew S and Goldstein, Tom and Studer, Christoph, ``MSE-optimal
neural network initialization via layer fusion", 2020 54th Annual Conference on
Information Sciences and Systems (CISS)

(c) 2020 raminaghods (rghods@cs.cmu.edu)

Preprocess the speech data -- only need to run this once to get the data_arrays

code for data processing from:
P. Warden, “Speech commands: A dataset for limited-vocabulary speech
recognition,” arXiv preprint: 1804.03209, Apr. 2018.
"""

from preprocess import load_dataset, save_data_to_array

seq_len = 128        # Number of steps, max of 128
n_channels = 11 # number of input channels maximum 11

DATA_PATH = "ENTER DATA PATH"
load_dataset(path=DATA_PATH,seq_len=seq_len)

# Save data to array file first
save_data_to_array(path=DATA_PATH,max_len=n_channels,seq_len=seq_len)
