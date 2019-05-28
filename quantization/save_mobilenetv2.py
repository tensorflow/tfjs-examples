# Copyright 2019 Google LLC. All Rights Reserved.
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

# This script saves MobileNetV2 in TensorFlow.js LayersModel format.

import os

import tensorflow as tf
import tensorflowjs as tfjs

MODEL_SAVE_PATH='models/MobileNetV2/original'


def main():
  model = tf.keras.applications.MobileNetV2(weights='imagenet')
  model_save_dir = os.path.dirname(MODEL_SAVE_PATH)
  if not os.path.isdir(model_save_dir):
    os.makedirs(model_save_dir)
  tfjs.converters.save_keras_model(model, MODEL_SAVE_PATH)
  print('Saved MobileNetV2 in tfjs_layer_model format at %s' % MODEL_SAVE_PATH)


if __name__ == '__main__':
  main()
