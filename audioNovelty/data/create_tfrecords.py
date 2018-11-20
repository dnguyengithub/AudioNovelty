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

"""
Creates a set of TFRecords from raw *.wav files.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import os
import random

import numpy as np
import librosa
import tensorflow as tf
from tqdm import tqdm

tf.app.flags.DEFINE_string("raw_wav_dir", "./datasets/",
                          "Directory containing TIMIT files.")
tf.app.flags.DEFINE_string("out_dir", "./datasets/",
                          "Output directory for TFRecord files.")
tf.app.flags.DEFINE_integer("duration", 3,
                            "Duration of the series (in second).")
tf.app.flags.DEFINE_float("valid_frac", 0.1,
                          "Fraction of train set to use as valid set. "
                          "Must be between 0.0 and 1.0.")

FLAGS = tf.app.flags.FLAGS
config = FLAGS

SAMPLING_RATE = 16000
SAMPLES_PER_TIMESTEP = 160
DURATION = FLAGS.duration

def get_filenames():
    """Get all wav filenames from the TIMIT archive."""
    files_train = sorted(glob.glob(os.path.join(FLAGS.raw_wav_dir,"train", "*.wav")))
    files_test = sorted(glob.glob(os.path.join(FLAGS.raw_wav_dir,"test", "*.wav")))
    files_train.sort(key=lambda f: int(filter(str.isdigit, f)))
    files_test.sort(key=lambda f: int(filter(str.isdigit, f)))
    return files_train, files_test

def load_wav(filename,mono):
    data, wav_rate = librosa.load(filename,
                              sr=SAMPLING_RATE,
                              mono=mono)
    # convert binaural to monaural
    if mono == False:
        data = np.mean(data,axis=0)
    # split
    l_wavs = []
    for d_i in range(0,int(len(data)/SAMPLING_RATE),DURATION):
        try:
            l_wavs.append(data[d_i*SAMPLING_RATE:(d_i+DURATION)*SAMPLING_RATE])
        except:
            continue
    return l_wavs, wav_rate


def preprocess(wavs, block_size, mean, std):
    """Normalize the wav data and reshape it into chunks."""
    processed_wavs = []
    for wav in tqdm(wavs):
        wav = (wav - mean) / std
        wav_length = wav.shape[0]
        if wav_length % block_size != 0:
            pad_width = block_size - (wav_length % block_size)
            wav = np.pad(wav, (0, pad_width), "constant")
        assert wav.shape[0] % block_size == 0
        wav = wav.reshape((-1, block_size))
        processed_wavs.append(wav)
    return processed_wavs


def create_tfrecord_from_wavs(wavs, output_file):
  """Writes processed wav files to disk as sharded TFRecord files."""
  with tf.python_io.TFRecordWriter(output_file) as builder:
    for wav in wavs:
      builder.write(wav.astype(np.float32).tobytes())

def main(unused_argv):
    l_filename_train, l_filename_test = get_filenames()

    print("Loading training *.wav files...")
    l_wav_train = []
    l_wav_test = []
    for f in tqdm(l_filename_train):
        l_wav_train_tmp, wav_rate = load_wav(f, mono=False)
        l_wav_train += l_wav_train_tmp
        
    print("Loading test *.wav files...")
    for f in tqdm(l_filename_test):
        l_wav_test_tmp, wav_rate = load_wav(f, mono=True)
        l_wav_test += l_wav_test_tmp
        
    random.seed(1234)
    random.shuffle(l_wav_train)
    valid_split_idx = int(0.1*len(l_wav_train))
    l_wav_valid = l_wav_train[:valid_split_idx]
    l_wav_train = l_wav_train[valid_split_idx:]
    

    # Calculate the mean and standard deviation of the train set.
    train_stacked = np.hstack(l_wav_train)
    train_mean = np.mean(train_stacked)
    train_std = np.std(train_stacked)
    print("train mean: %f  train std: %f" % (train_mean, train_std))

    # Process all data, normalizing with the train set statistics.
    print("Preprocessing...")
    processed_wav_train = preprocess(l_wav_train, 
                                SAMPLES_PER_TIMESTEP,
                                train_mean, train_std)
    processed_wav_valid = preprocess(l_wav_valid, 
                                SAMPLES_PER_TIMESTEP,
                                train_mean, train_std)
    processed_wav_test = preprocess(l_wav_test, 
                                SAMPLES_PER_TIMESTEP,
                                train_mean, train_std)
    # Write the datasets to disk.
    print("Writing to disk...")
    create_tfrecord_from_wavs(processed_wav_train,
                              os.path.join(FLAGS.out_dir, "train_{0}_{1}.tfrecord".format(DURATION,SAMPLES_PER_TIMESTEP)))
    create_tfrecord_from_wavs(processed_wav_valid,
                              os.path.join(FLAGS.out_dir, "valid_{0}_{1}.tfrecord".format(DURATION,SAMPLES_PER_TIMESTEP)))
    create_tfrecord_from_wavs(processed_wav_test,
                              os.path.join(FLAGS.out_dir, "test_{0}_{1}.tfrecord".format(DURATION,SAMPLES_PER_TIMESTEP)))


if __name__ == "__main__":
  tf.app.run()
