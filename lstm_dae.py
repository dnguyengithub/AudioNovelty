from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import logging
import os
import pickle

from audioNovelty import lstm_ae
from audioNovelty import bounds
from audioNovelty import smc

from audioNovelty.data import datasets
from audioNovelty.models import base
from audioNovelty.models import srnn
from audioNovelty.models import vrnn
import sonnet as snt

from tqdm import tqdm

from flags_config_lstm_dae import config
#config.dataset_path = "./datasets/tfrecords/train.tfrecord"
#config.mode = "train"

def main(unused_argv):
    fh = logging.FileHandler(os.path.join(config.logdir,config.log_filename+".log"))
    tf.logging.set_verbosity(tf.logging.INFO)
    # get TF logger
    logger = logging.getLogger('tensorflow')
    logger.addHandler(fh)
    if config.mode == "train":
        lstm_ae.run_train(config)
  
if __name__ == "__main__":
    tf.app.run(main)
