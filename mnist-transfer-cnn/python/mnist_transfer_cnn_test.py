# Copyright 2018 Google LLC. All Rights Reserved.
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
# =============================================================================
"""Test for the MNIST transfer learning CNN model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import shutil
import tempfile
import unittest

import numpy as np

from . import mnist_transfer_cnn

class MnistTest(unittest.TestCase):

  def setUp(self):
    self._tmp_dir = tempfile.mkdtemp()
    super(MnistTest, self).setUp()

  def tearDown(self):
    if os.path.isdir(self._tmp_dir):
      shutil.rmtree(self._tmp_dir)
    super(MnistTest, self).tearDown()

  def _getFakeMnistData(self):
    """Generate some fake MNIST data for testing."""
    x_train = np.random.rand(4, 28, 28)
    y_train = np.random.rand(4,)
    x_test = np.random.rand(4, 28, 28)
    y_test = np.random.rand(4,)
    return x_train, y_train, x_test, y_test

  def testWriteGte5DataToJsFiles(self):
    x_train, y_train, x_test, y_test = self._getFakeMnistData()
    js_file_prefix = os.path.join(self._tmp_dir, 'gte5')
    mnist_transfer_cnn.write_gte5_data(x_train, y_train, x_test, y_test,
                                       js_file_prefix)

    for js_path_suffix in ('.train.json', '.test.json'):
      with open(js_file_prefix + js_path_suffix, 'rt') as f:
        json_string = f.read()
        data = json.loads(json_string)
        if 'train' in js_path_suffix:
          self.assertEqual(x_train.shape[0], len(data))
        else:
          self.assertEqual(x_test.shape[0], len(data))
        self.assertEqual(28, len(data[0]['x']))
        self.assertEqual(28, len(data[0]['x'][0]))

  def testTrainWithFakeDataAndSave(self):
    x_train, y_train, x_test, y_test = self._getFakeMnistData()
    mnist_transfer_cnn.train_and_save_model(
        2, 2, 2, 2, 1, x_train, y_train, x_test, y_test,
        self._tmp_dir, optimizer='adam')

    # Check that the model json file is created.
    json.load(open(os.path.join(self._tmp_dir, 'model.json'), 'rt'))


if __name__ == '__main__':
  unittest.main()
