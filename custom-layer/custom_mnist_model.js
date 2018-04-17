/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
import * as tf from '@tensorflow/tfjs';

/**
 * This custom layer is similar to the 'relu' non-linear Activation `Layer`, but
 * it keeps both the negative and positive signal.  The input is centered at the
 * mean value, and then the negative activations and positive activations are
 * separated into different channels, meaning that there are twice as many
 * output channels as input channels.
 *
 * Implementing a custom `Layer` in general invovles specifying a `call`
 * function, and possibly also a `computeOutputShape` and `build` function. This
 * layer does not need a custom `build` function because it does not store any
 * variables.
 *
 * TODO(bileschi): File a github issue for the loading / saving of custom
 * layers.
 */
class Antirectifier extends tf.layers.Layer {
  constructor() {
    super({});
    // TODO(bileschi): Can we point to documentation on masking here?
    this.supportsMasking = true;
  }

  /**
   * This layer only works on 4D Tensors [batch, height, width, channels],
   * and produces output with twice as many channels.
   *
   * layer.computeOutputShapes must be overridden in the case that the output
   * shape is not the same as the input shape.
   * @param {*} inputShapes
   */
  computeOutputShape(inputShape) {
    return [inputShape[0], inputShape[1], inputShape[2], 2 * inputShape[3]]
  }

  /**
   * Centers the input and applies the following function to every element of
   * the input.
   *
   *     x => [max(x, 0), max(-x, 0)]
   *
   * The theory being that there may be signal in the both negative and positive
   * portions of the input.  Note that this will double the number of channels.
   * @param inputs Tensor to be treated.
   * @param kwargs Only used as a pass through to call hooks.  Unused in this
   *   example code.
   */
  call(inputs, kwargs) {
    this.invokeCallHook(inputs, kwargs);
    const origShape = inputs[0].shape;
    const flatShape =
        [origShape[0], origShape[1] * origShape[2] * origShape[3]];
    const flattened = inputs[0].reshape(flatShape);
    const centered = tf.sub(flattened, flattened.mean(1).expandDims(1));
    const pos = centered.relu().reshape(origShape);
    const neg = centered.neg().relu().reshape(origShape);
    return tf.concat([pos, neg], 3);
  }
}


/**
 * Creates and returns a Keras model with a custom Layer.
 */
export function customMnistModel() {
  // Set up the custom model using new Antirectifier instead of relu activation.
  const customModel = tf.sequential();
  customModel.add(tf.layers.conv2d({
    inputShape: [28, 28, 1],
    kernelSize: 5,
    filters: 8,
    strides: 1,
    activation: 'linear',
    kernelInitializer: 'varianceScaling'
  }));
  //
  // Here (and below) is the place that the custom layer is used.
  customModel.add(new Antirectifier());
  //
  customModel.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));
  customModel.add(tf.layers.conv2d({
    kernelSize: 5,
    filters: 16,
    strides: 1,
    activation: 'linear',
    kernelInitializer: 'varianceScaling'
  }));
  //
  // Here (and above) is the place that the custom layer is used.
  customModel.add(new Antirectifier());
  //
  customModel.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));
  customModel.add(tf.layers.flatten());
  customModel.add(tf.layers.dense({
    units: 10,
    kernelInitializer: 'varianceScaling',
    activation: 'softmax'
  }));
  return customModel;
}
