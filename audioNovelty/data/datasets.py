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

"""Code for creating sequence datasets.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle

import numpy as np
from scipy.sparse import coo_matrix
import tensorflow as tf

# The default number of threads used to process data in parallel.
DEFAULT_PARALLELISM = 12


def create_speech_dataset(path,
                          batch_size,
                          samples_per_timestep=25,
                          num_parallel_calls=DEFAULT_PARALLELISM,
                          prefetch_buffer_size=2048,
                          shuffle=False,
                          repeat=False):
  """Creates a speech dataset.

  Args:
    path: The path of a possibly sharded TFRecord file containing the data.
    batch_size: The batch size. If repeat is False then it is not guaranteed
      that the true batch size will match for all batches since batch_size
      may not necessarily evenly divide the number of elements.
    samples_per_timestep: The number of audio samples per timestep. Used to
      reshape the data into sequences of shape [time, samples_per_timestep].
      Should not change except for testing -- in all speech datasets 200 is the
      number of samples per timestep.
    num_parallel_calls: The number of threads to use for parallel processing of
      the data.
    prefetch_buffer_size: The size of the prefetch queues to use after reading
      and processing the raw data.
    shuffle: If true, shuffles the order of the dataset.
    repeat: If true, repeats the dataset endlessly.
  Returns:
    inputs: A batch of input sequences represented as a dense Tensor of shape
      [time, batch_size, samples_per_timestep]. The sequences in inputs are the
      sequences in targets shifted one timestep into the future, padded with
      zeros.
    targets: A batch of target sequences represented as a dense Tensor of
      shape [time, batch_size, samples_per_timestep].
    lens: An int Tensor of shape [batch_size] representing the lengths of each
      sequence in the batch.
  """
  filenames = [path]

  def read_speech_example(value):
    """Parses a single tf.Example from the TFRecord file."""
    decoded = tf.decode_raw(value, out_type=tf.float32)
    example = tf.reshape(decoded, [-1, samples_per_timestep])
    length = tf.shape(example)[0]
    return example, length

  # Create the dataset from the TFRecord files
  dataset = tf.data.TFRecordDataset(filenames).map(
      read_speech_example, num_parallel_calls=num_parallel_calls)
  dataset = dataset.prefetch(prefetch_buffer_size)

  if repeat: dataset = dataset.repeat()
  if shuffle: dataset = dataset.shuffle(prefetch_buffer_size)

  dataset = dataset.padded_batch(
      batch_size, padded_shapes=([None, samples_per_timestep], []))

  def process_speech_batch(data, lengths):
    """Creates Tensors for next step prediction."""
    data = tf.transpose(data, perm=[1, 0, 2])
    lengths = tf.to_int32(lengths)
    targets = data
    # Shift the inputs one step forward in time. Also remove the last timestep
    # so that targets and inputs are the same length.
    inputs = tf.pad(data, [[1, 0], [0, 0], [0, 0]], mode="CONSTANT")[:-1]
    # Mask out unused timesteps.
    inputs *= tf.expand_dims(
        tf.transpose(tf.sequence_mask(lengths, dtype=inputs.dtype)), 2)
    return inputs, targets, lengths

  dataset = dataset.map(process_speech_batch,
                        num_parallel_calls=num_parallel_calls)
  dataset = dataset.prefetch(prefetch_buffer_size)

  itr = dataset.make_one_shot_iterator()
  inputs, targets, lengths = itr.get_next()
  return inputs, targets, lengths
