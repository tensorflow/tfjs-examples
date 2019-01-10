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

"""Converts the IMDB dataset from .npz format into a JavaScript format.

Four files are generated as a result:
  - imdb_train_data.bin
  - imdb_train_targets.bin
  - imdb_test_data.bin
  - imdb_test_targets.bin

In the *data.bin files, the variable-length sentences are separated by the integer 1
which labels the beginning of the sentence.
All the word indices are stored as uint32 values.
In the *targets.bin files, the binary labels are stored as uint8 values.
"""

import argparse
import struct

from keras.datasets import imdb

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('out_prefix', type=str, help='Output path prefix')
  args = parser.parse_args()

  print('Loading imdb data via keras...')
  (x_train, y_train), (x_test, y_test) = imdb.load_data()

  for split in ('train', 'test'):
    data_path = '%s_%s_data.bin' % (args.out_prefix, split)
    xs = x_train if split == 'train' else x_test
    print('Writing data to file: %s' % data_path)
    with open(data_path, 'wb') as f:
      for sentence in xs:
        f.write(struct.pack('%di' % len(sentence), *sentence))

    targets_path = '%s_%s_targets.bin' % (args.out_prefix, split)
    ys = y_train if split == 'train' else y_test
    print('Writing targets to file: %s' % targets_path)
    with open(targets_path, 'wb') as f:
      f.write(struct.pack('%db' % len(ys), *ys))
