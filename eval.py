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
import scipy
import scipy.signal
import skimage.morphology

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


config.split="test"

THRESHOLD = config.anomaly_threshold
PEAK_THRESHOLD = config.peak_threshold
USE_CONTRARIO = config.use_contrario
CONTRARIO_EPS = config.contrario_eps
MAX_SEQUENCE_LEN = config.max_seq_len
MIN_SEGMENT_LEN = config.min_seg_len
PERCENTILE = config.percentile
USE_CORRECTION = config.use_correction
SUPER_PEAK_THRESHOLD = -900
SUPER_PEAK_ANOMALY_DURATION = 200 # 2s

SAMPLING_RATE = 16000
SMOOTH_SIZE = 21
PERCENTILE_FILTER_SIZE = config.filter_size
EROSION_SIZE = PERCENTILE_FILTER_SIZE
V_CORRELATION = np.ones(EROSION_SIZE)*1/float(EROSION_SIZE)


DURATION = 30

# LOG DIR
config.log_filename = config.bound+"-"\
                      +config.model+"-"\
                      +"latent_size"+"-"+str(config.latent_size)+"-"\
                      +os.path.basename(config.dataset_path)
config.logdir = os.path.join(config.log_dir,config.log_filename)
config.logdir = config.logdir.replace("test_","train_")

#config.dataset_path = config.dataset_path.replace("train",config.split)
config.dataset_path = "./datasets/test_{0}_160.tfrecord".format(DURATION)




m_label = np.load("./datasets/test/labels.npy")
savefile_dir = "./results/"+config.logdir.split("/")[-1]
if not os.path.exists(savefile_dir):
    os.mkdir(savefile_dir)
savefilename = os.path.join(savefile_dir,config.dataset_path.split("/")[-1]+"_result.pkl")




### Load or create
###############################################################################
if not config.rerun_graph:
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
        for inbatch_idx in range(tar_np.shape[1]):
                Tmp = dict()
                Tmp["data"] = tar_np[:,inbatch_idx,:]
                Tmp["ll"] = ll_np[inbatch_idx]
                Tmp["log_alphas"] = log_alphas_np[:,inbatch_idx] 
                l_Result.append(Tmp)
    
    ## DUMP
    if config.dump_result:
        with open(savefilename,"wb") as f:
            pickle.dump(l_Result,f)



v_time = np.arange(0,DURATION,float(config.data_dimension)/SAMPLING_RATE)
FIG_DPI = 150
TP = 0; FP = 0; TN = 0; FN = 0
d_idx = 20
for d_idx in tqdm(range(len(l_Result))):
#if False:
    Tmp = l_Result[d_idx]
    data = Tmp["data"]+0
    log_alphas = Tmp["log_alphas"]+0

    # Smooth
  
    max_signal = scipy.ndimage.filters.percentile_filter(log_alphas,
                                                                1,
                                                                size=PERCENTILE_FILTER_SIZE)

    percentile_signal = scipy.ndimage.filters.percentile_filter(log_alphas,
                                                                PERCENTILE,
                                                                size=PERCENTILE_FILTER_SIZE)
                                                                
    median_signal = scipy.signal.medfilt(log_alphas, kernel_size=3)
    mean_signal = np.correlate(log_alphas,V_CORRELATION,'same')

#    v_predict = (log_alphas[1:-1] < THRESHOLD)
    v_predict = (max_signal[0:-2] < THRESHOLD)
    v_peak = (max_signal[0:-2] < PEAK_THRESHOLD)
    v_label = (m_label[d_idx]==1)
    
    
    if USE_CONTRARIO:
        v_anomaly = contrario_utils.contrario_detection(v_predict,
                                                        v_peak = v_peak,
                                                        epsilon=CONTRARIO_EPS,
                                                        max_seq_len=MAX_SEQUENCE_LEN,
                                                        min_seg_len=MIN_SEGMENT_LEN,
                                                        erosion_size = int(PERCENTILE_FILTER_SIZE/2))
    else:
        v_anomaly = (percentile_signal[0:-2] < THRESHOLD)
        
    # Erosion
#    v_anomaly_eroded = skimage.morphology.binary_erosion(v_anomaly)
#    v_anomaly_eroded = (np.correlate(v_anomaly,V_CORRELATION,'same')==1)
    v_anomaly_eroded = scipy.ndimage.filters.percentile_filter(v_anomaly,
                                                        PERCENTILE,
                                                        size=PERCENTILE_FILTER_SIZE)

    v_anomaly_raw = v_anomaly_eroded+0 #copy


    ## REMOVE SHORT OR ZERO-PEAK ANOMALY 
    # The beginning of a abnormal sound is always a peak 
    l_anomaly_segments = contrario_utils.nonzero_segments(v_anomaly_eroded)
    for l_idx_anomaly_segment in l_anomaly_segments:
        if np.count_nonzero(log_alphas[0:-2][l_idx_anomaly_segment] < PEAK_THRESHOLD) < 1:
            v_anomaly_eroded[l_idx_anomaly_segment] = 0
        else: # There is at least one peak 
            d_idx_1st_peak = np.where(log_alphas[0:-2][l_idx_anomaly_segment] < PEAK_THRESHOLD)[0][0]
            v_anomaly_eroded[l_idx_anomaly_segment[0]:l_idx_anomaly_segment[0]+d_idx_1st_peak] = 0
    # No abnormal sound whose duration 0.6s
    l_anomaly_segments = contrario_utils.nonzero_segments(v_anomaly_eroded)
    for l_idx_anomaly_segment in l_anomaly_segments:
        if len(l_idx_anomaly_segment) < 60:
            v_anomaly_eroded[l_idx_anomaly_segment] = 0

    if USE_CORRECTION:
        ## SUPER PEAK CORRECTION
        # Whenever there is a super peak, there is a long abnormal sound.
        l_anomaly_segments = contrario_utils.nonzero_segments(v_anomaly_eroded)
        for l_idx_anomaly_segment in l_anomaly_segments:
            v_idx_super_peak = np.where(log_alphas[0:-2][l_idx_anomaly_segment] < SUPER_PEAK_THRESHOLD)
            if (v_idx_super_peak > 0) and (v_idx_super_peak < 4) :
                d_idx_begin = l_idx_anomaly_segment[0]+v_idx_super_peak[0]
                v_anomaly_eroded[d_idx_begin:d_idx_begin+SUPER_PEAK_ANOMALY_DURATION] = 1
            

    v_anomaly_raw = (v_anomaly_raw == 1) 
    v_anomaly_eroded = (v_anomaly_eroded == 1)
    
    ## UPDATE TP, FP, TN, FN
    TP += np.count_nonzero(v_anomaly_eroded[v_label])
    FP += np.count_nonzero(v_anomaly_eroded[~v_label])
    TN += np.count_nonzero(~v_anomaly_eroded[~v_label])
    FN += np.count_nonzero(~v_anomaly_eroded[v_label])


    if config.plot:
        plt.figure(figsize=(1920*2/FIG_DPI, 640*2/FIG_DPI), dpi=FIG_DPI) 

        plt.subplot(3,1,1)
        plt.plot(v_time[::20],data[::20,::5])
        plt.xlim([0,DURATION])
        plt.title("Contrario: {0}, Percentile: {1}, Filter size: {2}, Threshould: {3}, Max_seq_len: {4}, Eps: {5}".format(USE_CONTRARIO,
                                                                                                        PERCENTILE,
                                                                                                        PERCENTILE_FILTER_SIZE,
                                                                                                        THRESHOLD,
                                                                                                        MAX_SEQUENCE_LEN,
                                                                                                        CONTRARIO_EPS))
        
        plt.subplot(3,1,2)
        plt.plot(v_label,'r' )
#        plt.plot(v_predict,'y:')
        plt.plot(v_anomaly_raw,'b:')
        plt.plot(v_anomaly_eroded,'g-.')
#        plt.legend(["label","predict","anomaly_raw","anomaly"])
        plt.legend(["label","anomaly_raw","anomaly"])
        plt.xlim([0,len(v_label)])
        
        plt.subplot(3,1,3)
        plt.plot(log_alphas)
        plt.plot(max_signal,'r')
        plt.plot(percentile_signal,'k')
        plt.plot(mean_signal,'g')
        plt.plot(median_signal,'y')
        plt.legend(["log_prob","max","{0}-percentile".format(PERCENTILE),"mean","median_signal"])
        plt.plot(np.zeros_like(log_alphas)+THRESHOLD,'r')
        plt.plot(np.zeros_like(log_alphas)+PEAK_THRESHOLD,'k:')
        plt.xlim([0,len(log_alphas)])
        plt.ylim([-1500,500])
        
#        plt.show()

        figname = os.path.join(savefile_dir,config.dataset_path.split("/")[-1]+"_{0}_{1}_{2}_{3:03d}.png".format(MAX_SEQUENCE_LEN,
                                                                                                            PERCENTILE,
                                                                                                            PERCENTILE_FILTER_SIZE,
                                                                                                            d_idx))
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
    f.write("THRESHOLD = {0}, USE_CONTRARIO = {1}, CONTRARIO_EPS = {2}, MAX_SEQUENCE_LEN = {3}, MIN_SEGMENT_LEN = {4}, FILTER_SIZE = {5}".format(THRESHOLD,
                                                                                                                              USE_CONTRARIO,
                                                                                                                              CONTRARIO_EPS,
                                                                                                                              MAX_SEQUENCE_LEN,
                                                                                                                              MIN_SEGMENT_LEN,
                                                                                                                              PERCENTILE_FILTER_SIZE))
    f.write("\n")
    f.write("Precision: {0:02f}, Recall: {1:02f}, F1-score: {2:02f}".format(d_precision,
                                                                      d_recall,
                                                                      d_f1))
    f.write("\n")


