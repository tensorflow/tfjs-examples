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

"""Test for the Iris dataset module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest

import numpy as np

import iris_data

class IrisDataTest(unittest.TestCase):

  def testLoadData(self):
    iris_x, iris_y = iris_data.load()
    self.assertEqual(2, len(iris_x.shape))
    self.assertGreater(iris_x.shape[0], 0)
    self.assertEqual(4, iris_x.shape[1])
    self.assertEqual(iris_x.shape[0], iris_y.shape[0])
    self.assertEqual(3, iris_y.shape[1])
    self.assertTrue(
        np.allclose(np.ones([iris_y.shape[0], 1]), np.sum(iris_y, axis=1)))


if __name__ == '__main__':
  unittest.main()
