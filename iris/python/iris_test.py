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

"""Test for the Iris model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import shutil
import tempfile
import unittest

import iris


class IrisTest(unittest.TestCase):

  def setUp(self):
    self._tmp_dir = tempfile.mkdtemp()
    super(IrisTest, self).setUp()

  def tearDown(self):
    if os.path.isdir(self._tmp_dir):
      shutil.rmtree(self._tmp_dir)
    super(IrisTest, self).tearDown()

  def testTrainAndSaveNonSequential(self):
    final_train_accuracy = iris.train(100, self._tmp_dir)
    self.assertGreater(final_train_accuracy, 0.9)

    # Check that the model json file is created.
    json.load(open(os.path.join(self._tmp_dir, 'model.json'), 'rt'))

  def testTrainAndSaveSequential(self):
    final_train_accuracy = iris.train(100, self._tmp_dir, sequential=True)
    self.assertGreater(final_train_accuracy, 0.9)

    # Check that the model json file is created.
    json.load(open(os.path.join(self._tmp_dir, 'model.json'), 'rt'))


if __name__ == '__main__':
  unittest.main()
