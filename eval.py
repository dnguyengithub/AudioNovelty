from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import logging
import os
import pickle
import librosa
from tqdm import tqdm

from audioNovelty import runners
from audioNovelty import bounds
from audioNovelty import smc

from audioNovelty.data import datasets
from audioNovelty.models import base
from audioNovelty.models import srnn
from audioNovelty.models import vrnn

from audioNovelty.flags_config import config
config.num_samples = 1

config.split="test"
config.model = "vrnn"
config.latent_size=100

SAMPLING_RATE = 16000


# LOG DIR
config.log_filename = config.bound+"-"\
                      +config.model+"-"\
                      +"latent_size"+"-"+str(config.latent_size)+"-"\
                      +os.path.basename(config.dataset_path)
config.logdir = os.path.join(config.log_dir,config.log_filename)
#config.logdir = config.logdir.replace("test_","train_")

config.dataset_path = config.dataset_path.replace("train",config.split)
config.dataset_path = "./datasets/test30.tfrecord"

duration = int(filter(str.isdigit,config.dataset_path))

### Creates the evaluation graph
###############################################################################
if True:
    global_step = tf.train.get_or_create_global_step()
    inputs, targets, lengths, model, _ = runners.create_dataset_and_model(config, 
                                                                          split=config.split, 
                                                                          shuffle=False, 
                                                                          repeat=False)
    # Compute lower bounds on the log likelihood.
    # log_weights: A Tensor of shape [max_seq_len, batch_size, num_samples]
    #  containing the log weights at each timestep
    ll_per_seq, log_weights, _ = bounds.iwae(model, 
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

    l_Results = []
    d_key = 0
    
    while True:
        print(d_key)
        d_key+=1
        # tar_np.shape = (40, 8, 50) [seq_len, batch_size, data_size]
        # ll_np.shape = (), scalar, mean of the ll_per_t of all sequences in the batch 
        # log_weights: A Tensor of shape [max_seq_len, batch_size, num_samples]
        try:
            tar_np, ll_np, log_weights_np =\
                             sess.run([targets, ll_per_t, log_weights])
        except:
            break
        log_weights_np = np.squeeze(log_weights_np) #[seq_len, data_size], batch_size=1
        log_alphas_np = log_weights_np+0 # copy
        log_alphas_np[1:] = log_weights_np[1:]-log_weights_np[:-1] # decumulate
        for inbatch_idx in range(config.batch_size):
            Tmp = dict()
            Tmp["data"] = tar_np[:,inbatch_idx,:]
            Tmp["ll"] = ll_np[inbatch_idx]
            Tmp["log_alphas"] = log_alphas_np[:,inbatch_idx] 
            l_Results.append(Tmp)
            
    ## DUMP
    savefile_dir = "./results/"+config.logdir.split("/")[-1]
    if not os.path.exists(savefile_dir):
        os.mkdir(savefile_dir)
    savefilename = os.path.join(savefile_dir,config.dataset_path.split("/")[-1]+"_result.pkl")
    with open(savefilename,"wb") as f:
        pickle.dump(l_Results,f)




if False:
    v_time = np.arange(0,duration,1/80)
    FIG_DPI = 150
    for d_idx in tqdm(range(len(l_Results))):
        Tmp = l_Results[d_idx]
        data = Tmp["data"]+0
        log_alphas = Tmp["log_alphas"]+0
        
        plt.figure(figsize=(1920*2/FIG_DPI, 640*2/FIG_DPI), dpi=FIG_DPI) 
        plt.subplot(2,1,1)
        plt.plot(v_time,data)
        plt.xlim([0,duration])
    #    plt.yscale("log")
        plt.title("Waveform")
        plt.subplot(2,1,2)
        plt.plot(v_time,log_alphas)
        plt.plot(v_time,np.zeros_like(log_alphas)-500,'r')
        plt.xlim([0,duration])
        plt.ylim([-1500,500])
        plt.title("Log prob")
        figname = os.path.join(savefile_dir,config.dataset_path.split("/")[-1]+"_waveform_prob_{0:03d}.png".format(d_idx))
        plt.savefig(figname,dpi=FIG_DPI)
        plt.close()
    
    


#    for turn_idx in range(10):
#        tar_np, ll_np =  sess.run([targets, ll_per_t])
#        for inbatch_idx in range(config.batch_size):
#            data = tar_np[:,inbatch_idx,:]+0
#            data = data.reshape(-1)
#        #    plt.plot(data)
#        #    plt.show()
#            writefilename = "{0:02d}_{1:04d}_{2:02f}.wav".format(turn_idx,inbatch_idx, ll_np[inbatch_idx])
#            librosa.output.write_wav(writefilename, data, SAMPLING_RATE)

