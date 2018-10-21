#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 16:42:00 2018
@author: vnguye04
A set of utils for the a contrario anomaly detection.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import operator as op

P_ANOMALY = 0.1
H_P = 1/(1-2*P_ANOMALY)*np.log((1-P_ANOMALY)/P_ANOMALY)

def nCr(n, r):
    """Function calculates the number of combinations (n choose r)"""
    r = min(r, n-r)
    numer = reduce(op.mul, xrange(n, n-r, -1), 1)
    denom = reduce(op.mul, xrange(1, r+1), 1)
    return numer//denom

def nonzero_segments(x_):
    """Return list of consecutive nonzeros from x_"""
    run = []
    result = []
    for d_i in range(len(x_)):
        if x_[d_i] != 0:
            run.append(d_i)
        else:
            if len(run) != 0:
                result.append(run)
                run = []
    if len(run) != 0:
        result.append(run)
        run = []
    return result 
    
def zero_segments(x_):
    """Return list of consecutive zeros from x_"""
    run = []
    result = []
    for d_i in range(len(x_)):
        if x_[d_i] == 0:
            run.append(d_i)
        else:
            if len(run) != 0:
                result.append(run)
                run = []
    if len(run) != 0:
        result.append(run)
        run = []
    return result
    
def NFA(ns,k,d_sequence_len=None):
    """Number of False Alarms (unnormalized)"""
    B = 0
    for t in range(k,ns+1):
        B += nCr(ns,t)*(P_ANOMALY**t)*((1-P_ANOMALY)**(ns-t))
    if d_sequence_len is not None:
        return np.sum(np.arange(1,d_sequence_len+1))*B
    else:
        return B

def contrario_detection(v_A,
                        v_peak = None,
                        epsilon=10e-5,
                        max_seq_len=1000,
                        min_seg_len=1,
                        erosion_size=0):
    """
    A contrario detection algorithms
    INPUT:
        v_A_: abnormal point indicator vector
        epsilon: threshold
    OUTPUT:
        v_anomalies: abnormal segment indicator vector
        
    """
    if min_seg_len == 0:
        min_seg_len = 1
    if v_peak is None:
        v_peak = v_A + 0
    l_v_A_ = []
    l_v_peak_ = []
    l_v_anomaly = []
    for d_idx_split in range(0,len(v_A),max_seq_len):
        l_v_A_.append(v_A[d_idx_split:d_idx_split+max_seq_len]+0)
        l_v_peak_.append(v_peak[d_idx_split:d_idx_split+max_seq_len]+0)
    
    for d_idx_segment in range(len(l_v_A_)):
        v_A_ = l_v_A_[d_idx_segment]
        v_peak_ = l_v_peak_[d_idx_segment]
        d_sequence_len = len(v_A_)
        epsilon_ = epsilon/np.sum(np.arange(1,d_sequence_len+1)) # Divide by the number of posible sequences.
        v_anomaly = np.zeros(d_sequence_len)
        d_idx_anchor = 0
        v_raw_anomaly_tmp = v_A_ + 0
        
        while True:
            v_raw_anomaly_idx = np.where(v_raw_anomaly_tmp)[0]
            if len(v_raw_anomaly_idx) == 0:
                break
            d_idx_anchor += v_raw_anomaly_idx[0] 
            v_raw_anomaly_tmp = v_raw_anomaly_tmp[v_raw_anomaly_idx[0]:v_raw_anomaly_idx[-1]+1] # shrink
            
            d_segment_len = len(v_raw_anomaly_tmp)
            d_n_anomalies = int(np.count_nonzero(v_raw_anomaly_tmp))
            
            if (d_n_anomalies >= P_ANOMALY*d_segment_len + (1-P_ANOMALY)) and (d_segment_len >= min_seg_len): # Sanity check
                if NFA(d_segment_len,d_n_anomalies)<epsilon_: 
                    # peak check
                    v_seg_peak_tmp=v_peak_[d_idx_anchor:d_idx_anchor+d_segment_len]
                    if np.count_nonzero(v_seg_peak_tmp) >= 2*min_seg_len: # this segment is eps-meaningful
                        v_anomaly[d_idx_anchor+np.where(v_seg_peak_tmp)[0][0]:d_idx_anchor+d_segment_len] = 1 # the meaningful segment starts from the peak
                        # mark 1 with erosion
#                        if d_segment_len == d_segment_len: # Full segment is eps-meaningful, no erosion
#                            v_anomaly[d_idx_anchor:d_idx_anchor+d_segment_len] = 1
#                        else:
#                            if v_A_[0] == 1:
#                                v_anomaly[d_idx_anchor:d_idx_anchor+d_segment_len-erosion_size] = 1 # erode the right side only
#                            elif v_A_[-1] == 1:
#                                v_anomaly[d_idx_anchor+erosion_size:d_idx_anchor+d_segment_len] = 1 # erode the left side only
#                            else:
#                                v_anomaly[d_idx_anchor+erosion_size:d_idx_anchor+d_segment_len-erosion_size] = 1 # erode both sides

                        d_idx_anchor += len(v_raw_anomaly_tmp)
                        v_raw_anomaly_tmp = v_A_[d_idx_anchor:]+0
                        continue
            # if this segment is not eps-meaningful (either one of the ifs above is False)
            v_raw_nomaly_idx = np.where(v_raw_anomaly_tmp==0)[0]
            if len(v_raw_nomaly_idx) == 0:
                d_idx_anchor += len(v_raw_anomaly_tmp)
                v_raw_anomaly_tmp = v_A_[d_idx_anchor:]+0
                continue
            v_raw_anomaly_tmp = v_raw_anomaly_tmp[:v_raw_nomaly_idx[-1]+1]
        #end while
        l_v_anomaly.append(v_anomaly)
    
    return np.hstack(l_v_anomaly)

