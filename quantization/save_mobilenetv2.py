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