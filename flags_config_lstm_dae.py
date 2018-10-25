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

"""A script to define flags.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import os

# Shared flags.
tf.app.flags.DEFINE_enum("mode", "train",
                         ["train", "eval", "sample"],
                         "The mode of the binary.")
tf.app.flags.DEFINE_enum("model", "vrnn",
                         ["vrnn", "ghmm", "srnn"],
                         "Model choice.")
tf.app.flags.DEFINE_integer("latent_size", 216,
                            "The size of the latent state of the model.")
tf.app.flags.DEFINE_integer("num_layers", 3,
                            "Number of RNN layers.")
tf.app.flags.DEFINE_enum("dataset_type", "speech",
                         ["pianoroll", "speech", "pose"],
                         "The type of dataset.")
tf.app.flags.DEFINE_string("dataset_path", "./datasets/trainASF_3_54.tfrecord",
                           "Path to load the dataset from.")
tf.app.flags.DEFINE_integer("data_dimension", 54,
                            "The dimension of each vector in the data sequence. "
                            "Defaults to 88 for pianoroll datasets and 200 for speech "
                            "datasets. Should not need to be changed except for "
                            "testing.")
tf.app.flags.DEFINE_integer("sequence_length", 300,
                            "Sequence_length.")
tf.app.flags.DEFINE_integer("batch_size", 16,
                            "Batch size.")
tf.app.flags.DEFINE_integer("num_samples", 1,
                            "The number of samples (or particles) for multisample "
                            "algorithms.")
tf.app.flags.DEFINE_float("noise_std", 0.25,
                          "Noise std.")
tf.app.flags.DEFINE_float("keep_prob", 0.5,
                          "Keep probability.")
tf.app.flags.DEFINE_float("lambda_loss", 0.0015,
                          "Lambda loss amount.")
tf.app.flags.DEFINE_string("log_dir", "./chkpts",
                           "The directory to keep checkpoints and summaries in.")
tf.app.flags.DEFINE_integer("random_seed", None,
                            "A random seed for seeding the TensorFlow graph.")
tf.app.flags.DEFINE_integer("parallel_iterations", 30,
                            "The number of parallel iterations to use for the while "
                            "loop that computes the bounds.")

# Training flags.
tf.app.flags.DEFINE_enum("bound", "elbo",
                         ["elbo", "iwae", "fivo", "fivo-aux"],
                         "The bound to optimize.")
tf.app.flags.DEFINE_boolean("normalize_by_seq_len", True,
                            "If true, normalize the loss by the number of timesteps "
                            "per sequence.")
tf.app.flags.DEFINE_float("learning_rate", 0.0025,
                          "The learning rate for ADAM.")
tf.app.flags.DEFINE_integer("max_steps", int(1e9),
                            "The number of gradient update steps to train for.")
tf.app.flags.DEFINE_integer("summarize_every", 100,
                            "The number of steps between summaries.")
tf.app.flags.DEFINE_enum("resampling_type", "multinomial",
                         ["multinomial", "relaxed"],
                         "The resampling strategy to use for training.")
tf.app.flags.DEFINE_float("relaxed_resampling_temperature", 0.5,
                          "The relaxation temperature for relaxed resampling.")
tf.app.flags.DEFINE_enum("proposal_type", "filtering",
                         ["prior", "filtering", "smoothing",
                          "true-filtering", "true-smoothing"],
                         "The type of proposal to use. true-filtering and true-smoothing "
                         "are only available for the GHMM. The specific implementation "
                         "of each proposal type is left to model-writers.")

# Distributed training flags.
tf.app.flags.DEFINE_string("master", "",
                           "The BNS name of the TensorFlow master to use.")
tf.app.flags.DEFINE_integer("task", 0,
                            "Task id of the replica running the training.")
tf.app.flags.DEFINE_integer("ps_tasks", 0,
                            "Number of tasks in the ps job. If 0 no ps job is used.")
tf.app.flags.DEFINE_boolean("stagger_workers", True,
                            "If true, bring one worker online every 1000 steps.")

# Evaluation flags.
tf.app.flags.DEFINE_enum("split", "train",
                         ["train", "test", "valid"],
                         "Split to evaluate the model on.")

# Sampling flags.
tf.app.flags.DEFINE_integer("sample_length", 50,
                            "The number of timesteps to sample for.")
tf.app.flags.DEFINE_integer("prefix_length", 25,
                            "The number of timesteps to condition the model on "
                            "before sampling.")
tf.app.flags.DEFINE_string("sample_out_dir", None,
                           "The directory to write the samples to. "
                           "Defaults to logdir.")

# Eval flags
tf.app.flags.DEFINE_integer("upperbound", 50,
                            "upperbound")
tf.app.flags.DEFINE_integer("lowerbound", 50,
                            "upperbound")


# Solve tf >=1.8.0 flags bug
tf.app.flags.DEFINE_string('log_filename', '', 'log filename')
tf.app.flags.DEFINE_string('logdir', '', 'log directory')

FLAGS = tf.app.flags.FLAGS
config = FLAGS

# LOG DIR
config.log_filename = "lstm_dae"+"-"\
                      +config.model+"-"\
                      +"latent_size"+"-"+str(config.latent_size)+"-"\
                      +os.path.basename(config.dataset_path)
config.logdir = os.path.join(config.log_dir,config.log_filename)
config.logdir.replace("test_","train_")
if config.proposal_type != "filtering":
    config.logdir += "-" + config.proposal_type
if not os.path.exists(config.logdir):
    if config.mode == "train":
        os.mkdir(config.logdir)
    else:
        raise ValueError(config.logdir + " doesnt exist!")
