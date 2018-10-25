from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import logging
import os
import pickle

from audioNovelty import runners
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


_DEFAULT_INITIALIZERS = {"w": tf.contrib.layers.xavier_initializer(),
                        "b": tf.zeros_initializer()}
initializers = _DEFAULT_INITIALIZERS



def restore_checkpoint_if_exists(saver, sess, logdir):
    checkpoint = tf.train.get_checkpoint_state(logdir)
    if checkpoint:
        checkpoint_name = os.path.basename(checkpoint.model_checkpoint_path)
        full_checkpoint_path = os.path.join(logdir, checkpoint_name)
        saver.restore(sess, full_checkpoint_path)
        return True
    return False
    
def wait_for_checkpoint(saver, sess, logdir):
    while not restore_checkpoint_if_exists(saver, sess, logdir):
        tf.logging.info("Checkpoint not found in %s, sleeping for 60 seconds."
                    % logdir)
        time.sleep(60)

def create_logging_hook(step, loss_value):
    """Creates a logging hook that prints the bound value periodically."""
    loss_label = "loss"
    def summary_formatter(log_dict):
        return "Step %d, %s: %f" % (log_dict["step"], 
                                loss_label, 
                                log_dict["loss_value"])
    logging_hook = tf.train.LoggingTensorHook(
        {"step": step, "loss_value": loss_value},
        every_n_iter=config.summarize_every,
        formatter=summary_formatter)
    return logging_hook


def create_loss():
        
    ### Creates the training graph
    ###############################################################################

    inputs, targets, lengths, model, _ = runners.create_dataset_and_model(config, 
                                                                          split=config.split, 
                                                                          shuffle=False, 
                                                                          repeat=False)

    data_encoder = snt.nets.MLP(output_sizes=[config.latent_size],
                              initializers=initializers,
                              activation=tf.nn.softmax,
                              activate_final = True,
                              name="data_encoder")
    data_decoder = snt.nets.MLP(output_sizes=[config.data_dimension],
                              initializers=initializers,
                              activation=tf.nn.softmax,
                              activate_final = True,
                              name="data_decoder")

    ## tf Graph input
    #X = tf.placeholder("float", [None, timesteps, num_input])
    #Y = tf.placeholder("float", [None, num_classes])

    def encode_data(input_tensor, encoder):
        """Encodes or decode a timeseries of inputs with a time independent encoder.
        Args:
            inputs: A [time, batch, feature_dimensions] tensor.
            encoder: A network that takes a [batch, features_dimensions] input and
              encodes the input.
        Returns:
            A [time, batch, encoded_feature_dimensions] output tensor.
        """
        input_shape = tf.shape(input_tensor)
        num_timesteps, batch_size = input_shape[0], input_shape[1]
        reshaped_inputs = tf.reshape(input_tensor, [-1, input_tensor.shape[-1]])
        inputs_encoded = encoder(reshaped_inputs)
        inputs_encoded = tf.reshape(inputs_encoded,
                                      [num_timesteps, batch_size, encoder.output_size])
        return inputs_encoded


    def gaussian_noise_layer(input_tensor, std):
        noise = tf.random_normal(shape=tf.shape(input_tensor), mean=0.0, stddev=std, dtype=tf.float32) 
        return input_tensor + noise

    def BLSTM(_X, config):

        # Prepare data shape to match `rnn` function requirements
        # Current data input shape: (batch_size, timesteps, n_input)
        # Required shape: 'timesteps' tensors list of shape (batch_size, num_input)

        # Add Gaussian noise
        _X = gaussian_noise_layer(inputs, config.noise_std)
    #    _X = gaussian_noise_layer(_X, config.noise_std)
        
        # Encode
        _X = encode_data(_X, data_encoder)
        
        # Unstack to get a list of 'timesteps' tensors of shape (batch_size, num_input)
    #    _X = tf.unstack(_X, config.sequence_length, 1)

        output = _X
        for layer in range(config.num_layers):
            with tf.variable_scope('rnn_layer_{}'.format(layer),reuse=tf.AUTO_REUSE):

                cell_fw = tf.contrib.rnn.LSTMCell(config.latent_size, initializer=tf.truncated_normal_initializer(-0.1, 0.1, seed=2))
                cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, input_keep_prob = config.keep_prob)

                cell_bw = tf.contrib.rnn.LSTMCell(config.latent_size, initializer=tf.truncated_normal_initializer(-0.1, 0.1, seed=2))
                cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, input_keep_prob = config.keep_prob)

                outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw, 
                                                                  cell_bw, 
                                                                  output,
                                                                  dtype=tf.float32)
                output = tf.concat(outputs,2)

        return encode_data(output,data_decoder)

    reconstructed_signal = BLSTM(inputs, config)

    euclidean_loss = tf.reduce_mean(tf.square(targets - reconstructed_signal))

    # Loss, optimizer and evaluation
    l2 = config.lambda_loss* sum( tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables()
    ) # L2 loss prevents this overkill neural network to overfit the data
    loss = euclidean_loss + l2 # Softmax loss
    tf.summary.scalar("eclidean_loss", euclidean_loss)
    tf.summary.scalar("final_loss",loss)
    return loss

def create_graph():
    global_step = tf.train.get_or_create_global_step()
    loss = create_loss()
    opt = tf.train.AdamOptimizer(config.learning_rate)
    grads = opt.compute_gradients(loss, var_list=tf.trainable_variables())
    train_op = opt.apply_gradients(grads, global_step=global_step)
    return loss, train_op, global_step

device = tf.train.replica_device_setter(ps_tasks=config.ps_tasks)
with tf.Graph().as_default():
    if config.random_seed: 
        tf.set_random_seed(config.random_seed)
    with tf.device(device):
        
        loss, train_op, global_step = create_graph()
        log_hook = create_logging_hook(global_step, loss)
        start_training = not config.stagger_workers
        with tf.train.MonitoredTrainingSession(master=config.master,
                                            is_chief=config.task == 0,
                                            hooks=[log_hook],
                                            checkpoint_dir=config.logdir,
                                            save_checkpoint_secs=120,
                                            save_summaries_steps=config.summarize_every,
                                            log_step_count_steps=config.summarize_every) as sess:
            cur_step = -1
            while not sess.should_stop() and cur_step <= config.max_steps:
                if config.task > 0 and not start_training:
                    cur_step = sess.run(global_step)
                    tf.logging.info("task %d not active yet, sleeping at step %d" %
                            (config.task, cur_step))
                    time.sleep(30)
                    if cur_step >= config.task * 1000:
                        start_training = True
                else:
                    _, cur_step = sess.run([train_op, global_step])
