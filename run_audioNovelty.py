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

"""A script to run training for sequential latent variable models.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import logging
import os
from audioNovelty import runners

from audioNovelty.flags_config import config

def main(unused_argv):
  fh = logging.FileHandler(os.path.join(config.logdir,config.log_filename+".log"))
  tf.logging.set_verbosity(tf.logging.INFO)
  # get TF logger
  logger = logging.getLogger('tensorflow')
  logger.addHandler(fh)
  if config.model in ["vrnn", "srnn"]:
    if config.mode == "train":
      runners.run_train(config)
    elif config.mode == "eval":
      runners.run_eval(config)

if __name__ == "__main__":
  tf.app.run(main)
