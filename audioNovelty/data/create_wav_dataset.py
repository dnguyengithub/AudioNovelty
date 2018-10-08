# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Preprocesses TIMIT from raw wavfiles to create a set of TFRecords.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import os
import random
import re
import matplotlib.pyplot as plt

import numpy as np
import scipy.io.wavfile as scipywavefile
import librosa
import tensorflow as tf
from tqdm import tqdm

tf.app.flags.DEFINE_string("raw_wav_dir", "./datasets/train16kHz/",
                          "Directory containing TIMIT files.")
tf.app.flags.DEFINE_string("out_dir", "./datasets/traingset/",
                          "Output directory for wav files.")
tf.app.flags.DEFINE_string("out_prefix", "train",
                          "Output directory for wav files.")


FLAGS = tf.app.flags.FLAGS
config = FLAGS

SAMPLING_RATE = 16000
DURATION = 3 # in second


def get_filenames():
    """Get all wav filenames from the TIMIT archive."""
    files_train = sorted(glob.glob(os.path.join(FLAGS.raw_wav_dir, "*.wav")))
    
    return files

l_filenames = get_filenames()
filename = l_filenames[0]


def load_wav(filename):
#    wav_rate, data = scipywavefile.read(filename)
    data, wav_rate = librosa.load(filename,
                              sr=SAMPLING_RATE,
                              mono=False)
    # convert binaural to monaural
    data = np.mean(data,axis=0)
    # split
    l_wavs = []
    for d_i in range(0,int(5*60),DURATION):
        l_wavs.append(data[d_i*SAMPLING_RATE:(d_i+DURATION)*SAMPLING_RATE])
    return l_wavs, wav_rate


def main(unused_argv):
    l_filenames_train = get_filenames()

    print("Loading raw *.wav files...")
    l_wavs = []
    for f in tqdm(l_filenames):
        l_wav_tmp, wav_rate = load_wav(f)
        l_wavs += l_wav_tmp
    
    # write wav files
    if not os.path.exists(config.out_dir):
        os.makedirs(config.out_dir)
    print("Writing split *.wav files...")
    for d_i in tqdm(range(len(l_wavs))):
        writefilename = config.out_dir + config.out_prefix + "{0:04d}.wav".format(d_i)
#        scipywavefile.write(writefilename, SAMPLING_RATE, l_wavs[d_i])
        librosa.output.write_wav(writefilename, l_wavs[d_i], SAMPLING_RATE)
if __name__ == "__main__":
  tf.app.run()
  


"""
filename = "./datasets/traingset/train0000.wav"
data, wav_rate = librosa.load(filename,
                              sr=SAMPLING_RATE,
                              mono=True,
                              duration=DURATION)
"""
