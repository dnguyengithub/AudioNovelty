from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import logging
import os
import pickle

from grunt import runners
from grunt import bounds
from grunt import smc

from grunt.data import datasets
from grunt.models import base
from grunt.models import srnn
from grunt.models import vrnn

from grunt.flags_config import config
config.mode = "eval"
#config.model = "vrnn"
# LOG DIR
config.log_filename = config.bound+"-"\
                      +config.model+"-"\
                      +"latent_size"+"-"+str(config.latent_size)+"-"\
                      +os.path.basename(config.dataset_path)
config.logdir = os.path.join(config.log_dir,config.log_filename)
#config.logdir = config.logdir.replace("test_","train_")

config.dataset_path = config.dataset_path.replace("train_",config.split+"_")




### Creates the evaluation graph
###############################################################################
if True:
    global_step = tf.train.get_or_create_global_step()
    inputs, targets, lengths, model, _ = runners.create_dataset_and_model(config, 
                                                                          split=config.split, 
                                                                          shuffle=False, 
                                                                          repeat=False)
    # Compute lower bounds on the log likelihood.
    ll_per_seq, _, _ = bounds.iwae(model, 
                                        (inputs, targets),
                                        lengths, 
                                        num_samples=1,
                                        parallel_iterations=config.parallel_iterations)

#    elbo_ll = tf.reduce_sum(elbo_ll_per_seq)
#    batch_size = tf.shape(lengths)[0]
#    total_batch_length = tf.reduce_sum(lengths)
#    # Compute loss scaled by number of timesteps.
    ll_per_t = ll_per_seq / tf.to_float(lengths)
    batch_size = tf.shape(lengths)[0]
    
### Run evaluation
###############################################################################
if True:
    tf.Graph().as_default()
    if config.random_seed: tf.set_random_seed(config.random_seed)
    saver = tf.train.Saver()
    sess = tf.train.SingularMonitoredSession()
    runners.wait_for_checkpoint(saver, sess, config.logdir)

    total_tar_np = []
    total_ll_np = []
    
    while True:
        try:
            tar_np, ll_np =\
                             sess.run([targets, ll_per_t])
        except tf.errors.OutOfRangeError:
            break
        # tar_np.shape = (40, 8, 50) (seq_len, batch_size, data_size)
        # ll_np.shape = (), scalar, mean of the ll_per_t of all sequences in the batch 
        # batch_size_np = (), scalar = config.batch_size
        total_tar_np.append(tar_np)
        total_ll_np.append(ll_np)
    
    Result = dict()
    Result["data"] = np.hstack(total_tar_np)
    Result["ll"] = np.hstack(total_ll_np)
    savefile_dir = "./results/"+config.logdir.split("/")[-1]
    if not os.path.exists(savefile_dir):
        os.mkdir(savefile_dir)
    savefilename = os.path.join(savefile_dir,config.dataset_path.split("/")[-1]+".pkl")
    with open(savefilename,"wb") as f:
        pickle.dump(Result,f)
    
    
    
### Visualisation
###############################################################################
if False:
    d_N = tar_np.shape[1]
    plt.figure()
    for d_i in range(d_N):
        plt.subplot(d_N,1,d_i+1)
        tar = tar_np[:,d_i,:]+0
        plt.plot(tar.reshape(-1))
        plt.title(str(ll_np[d_i]))
        plt.axis("off")
    plt.show()

tmp = Result["data"][:,:10,:]+0
tmp = np.swapaxes(tmp,0,1)
plt.plot(tmp.reshape(-1)[2*3000:2*7000])
plt.show()
    

