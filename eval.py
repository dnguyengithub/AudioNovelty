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
from audioNovelty import contrario_utils

from audioNovelty.data import datasets
from audioNovelty.models import base
from audioNovelty.models import srnn
from audioNovelty.models import vrnn

from audioNovelty.flags_config import config
config.num_samples = 1


config.model = "vrnn"
config.latent_size=80
config.split="test"

THRESHOLD = config.eval_threshold
USE_CONTRARIO = config.use_contrario
CONTRARIO_EPS = config.contrario_eps
CONTRARIO_SIZE = 24
MAX_SEQUENCE_LEN = config.max_seq_len

SAMPLING_RATE = 16000
SMOOTH_SIZE = 3
V_CORRELATION = np.ones(SMOOTH_SIZE)*1/float(SMOOTH_SIZE)

# LOG DIR
config.log_filename = config.bound+"-"\
                      +config.model+"-"\
                      +"latent_size"+"-"+str(config.latent_size)+"-"\
                      +os.path.basename(config.dataset_path)
config.logdir = os.path.join(config.log_dir,config.log_filename)
config.logdir = config.logdir.replace("test_","train_")

#config.dataset_path = config.dataset_path.replace("train",config.split)
config.dataset_path = "./datasets/test_30_160.tfrecord"

duration = 30



m_label = np.load("./datasets/test/labels.npy")
savefile_dir = "./results/"+config.logdir.split("/")[-1]
if not os.path.exists(savefile_dir):
    os.mkdir(savefile_dir)
savefilename = os.path.join(savefile_dir,config.dataset_path.split("/")[-1]+"_result.pkl")




### Load or create
###############################################################################
if False:
#if os.path.exists(savefilename):
    print("Loading calculated Log probabilities...")
    with open(savefilename,'rb') as f:
        l_Result = pickle.load(f)
else:
    print("Creating and running the evaluation graph...")
    ### Creates the evaluation graph
    ###################################
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

    # Compute loss scaled by number of timesteps.
    ll_per_t = ll_per_seq / tf.to_float(lengths)
    batch_size = tf.shape(lengths)[0]

    tf.Graph().as_default()
    if config.random_seed: tf.set_random_seed(config.random_seed)
    saver = tf.train.Saver()
    sess = tf.train.SingularMonitoredSession()
    runners.wait_for_checkpoint(saver, sess, config.logdir)

    l_Result = []
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
            l_Result.append(Tmp)
    
    ## DUMP
#    with open(savefilename,"wb") as f:
#        pickle.dump(l_Result,f)



v_time = np.arange(0,duration,float(config.data_dimension)/SAMPLING_RATE)
FIG_DPI = 150
TP = 0; FP = 0; TN = 0; FN = 0
d_idx = 0
for d_idx in tqdm(range(len(l_Result))):
    Tmp = l_Result[d_idx]
    data = Tmp["data"]+0
    log_alphas = Tmp["log_alphas"]+0
#    v_predict = (log_alphas < -600)[1:-1]
    v_predict = (log_alphas[1:-1] < THRESHOLD)
    v_label = (m_label[d_idx]==1)
    
#    # A contrario detection
#    v_A = v_predict
#    v_anomaly = np.zeros(len(v_A))
#    
#    for d_i_4h in range(0,len(v_A)+1-CONTRARIO_SIZE):
#        v_A_4h = v_A[d_i_4h:d_i_4h+CONTRARIO_SIZE]
#        v_anomaly_i = contrario_utils.contrario_detection(v_A_4h,CONTRARIO_EPS)
#        v_anomaly[d_i_4h:d_i_4h+CONTRARIO_SIZE][v_anomalies_i==1] = 1
       
    v_anomaly = contrario_utils.contrario_detection(v_predict,epsilon=CONTRARIO_EPS,max_seq_len=MAX_SEQUENCE_LEN)
    
    v_anomaly = (v_anomaly == 1) 
    if USE_CONTRARIO:
        TP += np.count_nonzero(v_anomaly[v_label])
        FP += np.count_nonzero(v_anomaly[~v_label])
        TN += np.count_nonzero(~v_anomaly[~v_label])
        FN += np.count_nonzero(~v_anomaly[v_label])
    else:
        TP += np.count_nonzero(v_predict[v_label])
        FP += np.count_nonzero(v_predict[~v_label])
        TN += np.count_nonzero(~v_predict[~v_label])
        FN += np.count_nonzero(~v_predict[v_label])

    if config.plot:
        plt.figure(figsize=(1920*2/FIG_DPI, 640*2/FIG_DPI), dpi=FIG_DPI) 

        plt.subplot(3,1,1)
        plt.plot(v_time,data)
        plt.xlim([0,duration])
        
        plt.subplot(3,1,2)
        plt.plot(v_label,'r' )
        plt.plot(v_predict,'b:')
        plt.plot(v_anomaly,'g-.')
        plt.legend(["label","predict","contrario"])
        plt.xlim([0,len(v_label)])
        
        plt.subplot(3,1,3)
        plt.plot(v_time,log_alphas)
        plt.plot(v_time,np.zeros_like(log_alphas)+THRESHOLD,'r')
        plt.xlim([0,duration])
        plt.ylim([-1500,500])
        
    #    plt.show()
    #    
        figname = os.path.join(savefile_dir,config.dataset_path.split("/")[-1]+"_waveform_prob_{0:03d}.png".format(d_idx))
        plt.savefig(figname,dpi=FIG_DPI)
        plt.close()

d_precision = float(TP)/(TP+FP)
d_recall = float(TP)/(TP+FN)
d_f1 = 2*(d_precision*d_recall)/(d_precision+d_recall)
print("Precision: {0:02f}, Recall: {1:02f}, F1-score: {2:02f}".format(d_precision,
                                                                      d_recall,
                                                                      d_f1))
logfilename = os.path.join(savefile_dir,config.dataset_path.split("/")[-1]+"_log.txt")


import time
with open(logfilename,"ab+") as f:
    f.write("*************************************************")
    f.write("\n")
    f.write(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
    f.write("\n")
    f.write("THRESHOLD = -400, USE_CONTRARIO = True, CONTRARIO_EPS = 3e-4, MAX_SEQUENCE_LEN = 100")
    f.write("\n")
    f.write("Precision: {0:02f}, Recall: {1:02f}, F1-score: {2:02f}".format(d_precision,
                                                                      d_recall,
                                                                      d_f1))
    f.write("\n")

    

if False:
    v_time = np.arange(0,duration,float(config.data_dimension)/SAMPLING_RATE)
    FIG_DPI = 150
    for d_idx in tqdm(range(len(l_Result))):
        Tmp = l_Result[d_idx]
        data = Tmp["data"]+0
        log_alphas = Tmp["log_alphas"]+0
        
        plt.figure(figsize=(1920*2/FIG_DPI, 640*2/FIG_DPI), dpi=FIG_DPI) 
        plt.subplot(3,1,1)
        plt.plot(v_time,data)
        plt.xlim([0,duration])
    #    plt.yscale("log")
        plt.title("Waveform")
        plt.subplot(3,1,2)
        plt.plot(v_time,log_alphas)
        plt.plot(v_time,np.zeros_like(log_alphas)-THRESHOLD,'r')
        plt.xlim([0,duration])
        plt.ylim([-1500,500])
        plt.title("Log prob")
        
        plt.subplot(3,1,3)
        tmp = np.correlate(log_alphas, V_CORRELATION, 'valid')
        plt.plot(tmp)
        plt.plot(np.zeros_like(tmp)-THRESHOLD,'r')
        plt.xlim([0,len(tmp)])
        plt.ylim([-1500,500])
        plt.title("Smoothed log prob")
        
        figname = os.path.join(savefile_dir,config.dataset_path.split("/")[-1]+"_waveform_prob_{0:03d}.png".format(d_idx))
        plt.savefig(figname,dpi=FIG_DPI)
        plt.close()
    


if False:
    
    plt.figure()
    
    plt.subplot(5,1,1)
    plt.plot(v_time,data)
    plt.xlim([0,duration])
    
    plt.subplot(5,1,2)
    plt.plot(v_label)
    plt.xlim([0,len(v_label)])
    
    plt.subplot(5,1,3)
    plt.plot(v_time,log_alphas)
    plt.plot(v_time,np.zeros_like(log_alphas)+THRESHOLD,'r')
    plt.xlim([0,duration])
    plt.ylim([-1500,500])
    
    plt.subplot(5,1,4)
    plt.plot(v_anomalies)
    plt.xlim([0,len(v_label)])
    
    plt.subplot(5,1,5)
    plt.plot(v_predict)
    plt.xlim([0,len(v_label)])
    plt.show()

"""
acc = np.count_nonzero(v_predict==v_label)/len(v_label)
print(acc)

import sklearn.metrics
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.utils.fixes import signature

y_test = m_label[d_idx]+0
y_score = -Tmp["log_alphas"][:-2]
average_precision = sklearn.metrics.average_precision_score(y_test, y_score)

print('Average precision-recall score: {0:0.2f}'.format(average_precision))
      



precision, recall, _ = precision_recall_curve(y_test, y_score)

# In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
step_kwargs = ({'step': 'post'}
               if 'step' in signature(plt.fill_between).parameters
               else {})
plt.step(recall, precision, color='b', alpha=0.2,
         where='post')
plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
          average_precision))
plt.show()

"""


#    for turn_idx in range(10):
#        tar_np, ll_np =  sess.run([targets, ll_per_t])
#        for inbatch_idx in range(config.batch_size):
#            data = tar_np[:,inbatch_idx,:]+0
#            data = data.reshape(-1)
#        #    plt.plot(data)
#        #    plt.show()
#            writefilename = "{0:02d}_{1:04d}_{2:02f}.wav".format(turn_idx,inbatch_idx, ll_np[inbatch_idx])
#            librosa.output.write_wav(writefilename, data, SAMPLING_RATE)

