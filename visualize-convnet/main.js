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

/**
 * Based on
 * https://github.com/fchollet/deep-learning-with-python-notebooks/blob/master/5.4-visualizing-what-convnets-learn.ipynb
 */

const argparse = require('argparse');
const fs = require('fs');
const path = require('path');
const shelljs = require('shelljs');
const tf = require('@tensorflow/tfjs');
const utils = require('./utils');
const imagenetClasses = require('./imagenet_classes');

const EPSILON = 1e-5;  // "Fudge" factor to prevent division by zero.

// TODO(cais): Deduplicate with index.js.
/**
 * Generate the maximally-activating input image for a conv2d layer filter.
 *
 * Uses gradient ascent in input space.
 *
 * @param {tf.Model} model The model that the conv2d layer of interest belongs
 *   to.
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
    // model but the output is the convolutional layer of interest.
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
        const norm = tf.sqrt(tf.mean(tf.square(grads))).add(EPSILON);
        // Important trick: scale the gradient with the magnitude (norm)
        // of the gradient.
        return grads.div(norm);
      });
      // Perform one step of gradient ascent: Update the image along the
      // direction of the gradient.
      image = image.add(scaledGrads);
    }
    return deprocessImage(image);
  });
}

function deprocessImage(x) {
  return tf.tidy(() => {
    const {mean, variance} = tf.moments(x);
    x = x.sub(mean);
    x = x.div(tf.sqrt(variance).add(EPSILON));
    // Clip to [0, 1].
    x = x.add(0.5);
    x = tf.clipByValue(x, 0, 1);
    x = x.mul(255);
    return tf.clipByValue(x, 0, 255).asType('int32');
  });
}

/**
 * Calcuate and save the maximally-activating input images for a covn2d layer.
 *
 * @param {tf.Model} model The model that the conv2d layer of interest belongs
 *   to.
 * @param {string} layerName The name of the layer of interest.
 * @param {number} numFilters Number of the conv2d layer's filter to calculate
 *   maximally-activating inputs for. If this exceeds the number of filters
 *   that the conv2d layer has, it will be cut off.
 * @param {string} outputDir Path to the directory to which the output image
 *   will be written.
 * @returns {string[]} Paths to the image files generated in this call.
 */
async function writeConvLayerFilters(
    model, layerName, numFilters, iterations, outputDir) {
  const filePaths = [];
  const maxFilters = model.getLayer(layerName).getWeights()[0].shape[3];
  if (numFilters > maxFilters) {
    numFilters = maxFilters;
  }
  for (let i = 0; i < numFilters; ++i) {
    console.log(
        `Processing layer ${layerName}, filter ${i + 1} of ${numFilters}`);
    const imageTensor = inputGradientAscent(model, layerName, i, iterations);
    const outputFilePath = path.join(outputDir, `${layerName}_${i + 1}.png`);
    filePaths.push(outputFilePath);
    await utils.writeImageTensorToFile(imageTensor, outputFilePath);
    imageTensor.dispose();
    console.log(`  --> ${outputFilePath}`);
  }
  return filePaths;
}

async function writeInternalActivationAndGetOutput(
    model, layerNames, inputImage, filters, outputDir) {
  const layerName2FilePaths = {};
  const layerOutputs =
      layerNames.map(layerName => model.getLayer(layerName).output);
  const compositeModel = tf.model(
      {inputs: model.input, outputs: layerOutputs.concat(model.outputs[0])});
  const outputs = compositeModel.predict(inputImage);
  for (let i = 0; i < outputs.length - 1; ++i) {
    const layerName = layerNames[i];
    const activationTensors =
        tf.split(outputs[i], outputs[i].shape[outputs[i].shape.length - 1], -1);
    const actualFilters = filters <= activationTensors.length ?
        filters :
        activationTensors.length;
    const filePaths = [];
    for (let j = 0; j < actualFilters; ++j) {
      const imageTensor = tf.tidy(
          () => deprocessImage(tf.tile(activationTensors[j], [1, 1, 1, 3])));
      const outputFilePath = path.join(outputDir, `${layerName}_${j + 1}.png`);
      filePaths.push(outputFilePath);
      await utils.writeImageTensorToFile(imageTensor, outputFilePath);
    }
    layerName2FilePaths[layerName] = filePaths;
    tf.dispose(activationTensors);
  }
  tf.dispose(outputs.slice(0, outputs.length - 1));
  return {modelOutput: outputs[outputs.length - 1], layerName2FilePaths};
}

function parseArguments() {
  const parser =
      new argparse.ArgumentParser({description: 'Visualize convnet'});
  parser.addArgument('modelJsonUrl', {
    type: 'string',
    help: 'URL to model JSON. Can be a file://, http://, or https:// URL'
  });
  parser.addArgument('convLayerNames', {
    type: 'string',
    help: 'Names of the conv2d layers to visualize, separated by commas ' +
        'e.g., (block1_conv1,block2_conv1,block3_conv1,block4_conv1)'
  });
  parser.addArgument('--inputImage', {
    type: 'string',
    defaultValue: '',
    help: 'Path to the input image. If specified, will compute the internal' +
        'activations of the specified convolutional layers. If not specified, ' +
        'will compute the maximally-activating input images using gradient ascent.'
  });
  parser.addArgument('--outputDir', {
    type: 'string',
    defaultValue: 'dist/filters',
    help: 'Output directory to which the image files and the manifest will ' +
        'be written'
  });
  parser.addArgument('--filters', {
    type: 'int',
    defaultValue: 64,
    help: 'Number of filters to visualize for each conv2d layer'
  });
  parser.addArgument('--iterations', {
    type: 'int',
    defaultValue: 40,
    help: 'Number of iterations to use for gradient ascent'
  });
  parser.addArgument(
      '--gpu',
      {action: 'storeTrue', help: 'Use tfjs-node-gpu (required CUDA GPU).'});
  return parser.parseArgs();
}

async function run() {
  const args = parseArguments();
  if (args.gpu) {
    // Use GPU bindings.
    require('@tensorflow/tfjs-node-gpu');
  } else {
    // Use CPU bindings.
    require('@tensorflow/tfjs-node');
  }

  console.log('Loading model...');
  if (args.modelJsonUrl.indexOf('http://') === -1 &&
      args.modelJsonUrl.indexOf('https://') === -1 &&
      args.modelJsonUrl.indexOf('file://') === -1) {
    args.modelJsonUrl = `file://${args.modelJsonUrl}`;
  }
  const model = await tf.loadModel(args.modelJsonUrl);
  console.log('Model loading complete.');

  if (!fs.existsSync(args.outputDir)) {
    shelljs.mkdir('-p', args.outputDir);
  }

  if (args.inputImage != null && args.inputImage !== '') {
    // Compute the internal activations of the conv layers' outputs.
    const imageHeight = model.inputs[0].shape[1];
    const imageWidth = model.inputs[0].shape[2];
    const x = await utils.readImageTensorFromFile(
        args.inputImage, imageHeight, imageWidth);
    const layerNames = args.convLayerNames.split(',');
    const {modelOutput, layerName2FilePaths} =
        await writeInternalActivationAndGetOutput(
            model, layerNames, x, args.filters, args.outputDir);

    const topNum = 10;
    const {values: topKVals, indices: topKIndices} =
        tf.topk(modelOutput, topNum);
    const values = await topKVals.data();
    const indices = await topKIndices.data();
    const manifest = {indices, values, layerName2FilePaths};

    console.log(`Top-${topNum} classes:`)
    for (let i = 0; i < topNum; ++i) {
      console.log(
          `  ${imagenetClasses.IMAGENET_CLASSES[indices[i]]}: ` +
          `${values[i].toFixed(4)}`);
    }

    const manifestPath = path.join(args.outputDir, 'activation-manifest.json');
    fs.writeFileSync(manifestPath, JSON.stringify(manifest));
  } else {
    // Calculate the maximally-activating input images for the conv layers'
    // filters.
    const layerNames = args.convLayerNames.split(',');
    const manifest = {layers: []};
    for (let i = 0; i < layerNames.length; ++i) {
      const layerName = layerNames[i];
      console.log(
          `\n=== Processing layer ${i + 1} of ${layerNames.length}: ` +
          `${layerName} ===`);
      const filePaths = await writeConvLayerFilters(
          model, layerName, args.filters, args.iterations, args.outputDir);
      manifest.layers.push({layerName, filePaths});
    }
    // Write manifest to file.
    const manifestPath = path.join(args.outputDir, 'filters-manifest.json');
    fs.writeFileSync(manifestPath, JSON.stringify(manifest));
  }
};

run();
