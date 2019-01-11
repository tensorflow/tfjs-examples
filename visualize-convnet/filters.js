/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
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

/**
 * Algorithms for analyzing and visualizing the convolutional filters
 * internal to a convnet.
 *
 * 1. Retrieving internal activations of a convnet.
 *    See function `writeInternalActivationAndGetOutput()`.
 * 2. Calculate maximally-activating input image for convnet filters, using
 *    gradient ascent in input space.
 *    See function `inputGradientAscent()`.
 **/

const path = require('path');
const tf = require('@tensorflow/tfjs');
const utils = require('./utils');

/**
 * Write internal activation of conv layers to file; Get model output.
 *
 * @param {tf.Model} model The model of interest.
 * @param {string[]} layerNames Names of layers of interest.
 * @param {tf.Tensor4d} inputImage The input image represented as a 4D tensor
 *   of shape [1, height, width, 3].
 * @param {number} numFilters Number of filters to run for each convolutional
 *   layer. If it exceeds the number of filters of a convolutional layer, it
 *   will be cut off.
 * @param {string} outputDir Path to the directory to which the image files
 *   representing the activation will be saved.
 * @return modelOutput: final output of the model as a tf.Tensor.
 *         layerName2FilePaths: an object mapping layer name to the paths to the
 *           image files saved for the layer's activation.
 *         layerName2FilePaths: an object mapping layer name to the height
 *           and width of the layer's filter outputs.
 */
async function writeInternalActivationAndGetOutput(
    model, layerNames, inputImage, numFilters, outputDir) {
  const layerName2FilePaths = {};
  const layerName2ImageDims = {};
  const layerOutputs =
      layerNames.map(layerName => model.getLayer(layerName).output);

  // Construct a model that returns all the desired internal activations,
  // in addition to the final output of the original model.
  const compositeModel = tf.model(
      {inputs: model.input, outputs: layerOutputs.concat(model.outputs[0])});

  // `outputs` is an array of `tf.Tensor`s consisting of the internal-activation
  // values and the final output value.
  const outputs = compositeModel.predict(inputImage);

  for (let i = 0; i < outputs.length - 1; ++i) {
    const layerName = layerNames[i];
    // Split the activation of the convolutional layer by filter.
    const activationTensors =
        tf.split(outputs[i], outputs[i].shape[outputs[i].shape.length - 1], -1);
    const actualNumFilters = numFilters <= activationTensors.length ?
        numFilters :
        activationTensors.length;
    const filePaths = [];
    let imageTensorShape;
    for (let j = 0; j < actualNumFilters; ++j) {
      // Format activation tensors and write them to disk.
      const imageTensor = tf.tidy(
          () => deprocessImage(tf.tile(activationTensors[j], [1, 1, 1, 3])));
      const outputFilePath = path.join(outputDir, `${layerName}_${j + 1}.png`);
      filePaths.push(outputFilePath);
      await utils.writeImageTensorToFile(imageTensor, outputFilePath);
      imageTensorShape = imageTensor.shape;
    }
    layerName2FilePaths[layerName] = filePaths;
    layerName2ImageDims[layerName] = imageTensorShape.slice(1, 3);
    tf.dispose(activationTensors);
  }
  tf.dispose(outputs.slice(0, outputs.length - 1));
  return {
    modelOutput: outputs[outputs.length - 1],
    layerName2FilePaths,
    layerName2ImageDims
  };
}


/**
 * Generate the maximally-activating input image for a conv2d layer filter.
 *
 * Uses gradient ascent in input space.
 *
 * @param {tf.Model} model The model that the convolutional layer of interest
 *   belongs to.
 * @param {string} layerName Name of the convolutional layer.
 * @param {number} filterIndex Index to the filter of interest. Must be
 *   < number of filters of the conv2d layer.
 * @param {number} iterations Number of gradient-ascent iterations.
 * @return {tf.Tensor} The maximally-activating input image as a tensor.
 */
function inputGradientAscent(model, layerName, filterIndex, iterations = 40) {
  return tf.tidy(() => {
    const imageH = model.inputs[0].shape[1];
    const imageW = model.inputs[0].shape[2];
    const imageDepth = model.inputs[0].shape[3];

    // Create an auxiliary model of which input is the same as the original
    // model but the output is the output of the convolutional layer of
    // interest.
    const layerOutput = model.getLayer(layerName).output;
    const auxModel = tf.model({inputs: model.inputs, outputs: layerOutput});

    // This function calculates the value of the convolutional layer's
    // output at the designated filter index.
    const lossFunction = (input) =>
        auxModel.apply(input, {training: true}).gather([filterIndex], 3);

    // This function (`gradient`) calculates the gradient of the convolutional
    // filter's output with respect to the input image.
    const gradients = tf.grad(lossFunction);

    // Form a random image as the starting point of the gradient ascent.
    let image = tf.randomUniform([1, imageH, imageW, imageDepth], 0, 1)
                    .mul(20)
                    .add(128);

    for (let i = 0; i < iterations; ++i) {
      const scaledGrads = tf.tidy(() => {
        const grads = gradients(image);
        const norm =
            tf.sqrt(tf.mean(tf.square(grads))).add(tf.ENV.get('EPSILON'));
        // Important trick: scale the gradient with the magnitude (norm)
        // of the gradient.
        return grads.div(norm);
      });
      // Perform one step of gradient ascent: Update the image along the
      // direction of the gradient.
      image = tf.clipByValue(image.add(scaledGrads), 0, 255);
    }
    return deprocessImage(image);
  });
}

/** Center and scale input image so the pixel values fall into [0, 255]. */
function deprocessImage(x) {
  return tf.tidy(() => {
    const {mean, variance} = tf.moments(x);
    x = x.sub(mean);
    // Add a small positive number (EPSILON) to the denominator to prevent
    // division-by-zero.
    x = x.div(tf.sqrt(variance).add(tf.ENV.get('EPSILON')));
    // Clip to [0, 1].
    x = x.add(0.5);
    x = tf.clipByValue(x, 0, 1);
    x = x.mul(255);
    return tf.clipByValue(x, 0, 255).asType('int32');
  });
}

module.exports = {
  inputGradientAscent,
  writeInternalActivationAndGetOutput
};
