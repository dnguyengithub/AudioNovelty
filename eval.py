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
#config.mode = "eval"
#config.model = "srnn"
#config.latent_size=100


# LOG DIR
config.log_filename = config.bound+"-"\
                      +config.model+"-"\
                      +"latent_size"+"-"+str(config.latent_size)+"-"\
                      +os.path.basename(config.dataset_path)
config.logdir = os.path.join(config.log_dir,config.log_filename)
#config.logdir = config.logdir.replace("test_","train_")

config.dataset_path = config.dataset_path.replace("train",config.split)




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
    # m_data: [samples, mel_t, mel_f]
    # v_ll: [samples]
    m_data = np.swapaxes(np.hstack(total_tar_np),0,1) 
    v_ll = np.hstack(total_ll_np)
    Result["data"] = m_data
    Result["ll"] = v_ll
    savefile_dir = "./results/"+config.logdir.split("/")[-1]
    if not os.path.exists(savefile_dir):
        os.mkdir(savefile_dir)
    savefilename = os.path.join(savefile_dir,config.dataset_path.split("/")[-1]+".pkl")
    with open(savefilename,"wb") as f:
        pickle.dump(Result,f)
    
d_mean = np.mean(Result["ll"])
d_std = np.std(Result["ll"])
print(config.dataset_path)
print("Mean: ",d_mean)
print("Std: ",d_std)



## LL scatter plot
FIG_DPI = 150
plt.figure(figsize=(1920*2/FIG_DPI, 640*2/FIG_DPI), dpi=FIG_DPI) 
plt.plot(Result["ll"],'o')
plt.title(config.dataset_path+", mean = "+str(d_mean)+", std = "+str(d_std))
figname = savefilename.replace(".pkl",".png")
plt.savefig(figname,dpi=FIG_DPI)
plt.show()

## Mel Spectrogram and ll plot
###############################################################################
d_N = 6
bad = []
ll_bad = []
good = []
ll_good = []
for d_i in range(d_N):
    try:
        bad.append(m_data[v_ll<(d_mean-1*d_std)][d_i]+0)
        ll_bad.append(v_ll[v_ll<(d_mean-1*d_std)][d_i]+0)
    except:
        continue
    try:
        good.append(m_data[v_ll>(d_mean+0.5*d_std)][d_i]+0)
        ll_good.append(v_ll[v_ll>(d_mean+0.5*d_std)][d_i]+0)
    except:
        continue
FIG_DPI = 150
plt.figure(figsize=(1920*2/FIG_DPI, 640*2/FIG_DPI), dpi=FIG_DPI) 
plt.title("YYYYYYY")
for d_i in range(d_N):
    try:
        plt.subplot(2,d_N,d_i+1)
        plt.imshow(np.swapaxes(good[d_i],0,1))
        plt.title("Good, "+str(int(ll_good[d_i])))
        plt.axis("off")
        plt.colorbar()
    except:
        continue
    try:
        plt.subplot(2,d_N,d_i+1+d_N)
        plt.imshow(np.swapaxes(bad[d_i],0,1))
        plt.title("Bad, "+str(int(ll_bad[d_i])))
        plt.axis("off")
        plt.colorbar()
    except:
        continue
figname = savefilename.replace(".pkl","_examples.png")
plt.savefig(figname,dpi=FIG_DPI)
plt.show()




    

