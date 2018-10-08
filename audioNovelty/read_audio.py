
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import os
import random
import re
#import sys
#sys.path.append("..")

import numpy as np
import tensorflow as tf
import wave
import matplotlib.pyplot as plt
import scipy
import librosa
import librosa.display
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', help='filename', default='sel.01.Grunt.wav')
    args = parser.parse_args()
    
    
    filename = os.path.join("./data/raw/",args.filename)
    wav_rate, data = scipy.io.wavfile.read(filename)
    wav = data[:,0]


    t_begin = 26
    t_duration = 10
    x_coords = np.arange(10*wav_rate)/wav_rate+t_begin
    y, sr = librosa.load(filename,
                         sr=wav_rate,
                         mono=False,
                         offset=26,
                         duration=10,
                         dtype=np.float16)





    x_axis = np.arange(int(t_begin*wav_rate),int(t_end*wav_rate))/float(wav_rate)
    plt.subplot(2,1,1)
    plt.plot(x_axis,data[int(t_begin*wav_rate):int(t_end*wav_rate),0])
    plt.subplot(2,1,2)
    plt.plot(x_axis,data[int(t_begin*wav_rate):int(t_end*wav_rate),0])
    plt.show()

    plt.subplot(2,1,1)
    plt.plot(data[:,0])
    plt.subplot(2,1,2)
    plt.plot(y[0,:])
    plt.show()

    hop_length = 256
    S = librosa.feature.melspectrogram(y[0,:], 
                                       sr=sr,
                                       n_fft=2048*2,
                                       hop_length=hop_length, 
                                       n_mels=256,
                                       fmax=1000)
    S = S[:,:-1]
    plt.figure()
    S_db = librosa.power_to_db(S,ref=np.max)
    librosa.display.specshow(S_db, 
                             fmax=500,
                             y_axis='mel',
                             x_axis='time',
                             x_coords=x_coords[::hop_length])
    plt.show()

plt.imshow(S_db)
plt.show()


