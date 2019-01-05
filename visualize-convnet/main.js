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
 *
 * This script performs the following operations:
 * 1. Retrieving internal activations of a convnet.
 *    See function `writeInternalActivationAndGetOutput`.
 * 2. Calculate maximally-activating input image for convnet filters, using
 *    gradient ascent in input space.
 *    See function `inputGradientAscent`.
 * 3. Get visual interpretation of which parts of the image more most
 *    responsible for a convnet's classification decision, using the
 *    gradient-based class activation map (CAM) method.
 *    See function `gradClassActivationMap`.
 */

const argparse = require('argparse');
const fs = require('fs');
const path = require('path');
const shelljs = require('shelljs');
const tf = require('@tensorflow/tfjs');
const utils = require('./utils');
const imagenetClasses = require('./imagenet_classes');

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

function deprocessImage(x) {
  return tf.tidy(() => {
    const {mean, variance} = tf.moments(x);
    x = x.sub(mean);
    x = x.div(tf.sqrt(variance).add(tf.ENV.get('EPSILON')));
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

/**
 * Write internal activation of conv layers to file; Get model output.
 *
 * @param {tf.Model} model The model of interest.
 * @param {string[]} layerNames Names of layers of interest.
 * @param {tf.Tensor4d} inputImage The input image represented as a 4D tensor
 *   of shape [1, height, width, 3].
 * @param {number} filters Number of filters to run for each convolutional
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
    model, layerNames, inputImage, filters, outputDir) {
  const layerName2FilePaths = {};
  const layerName2ImageDims = {};
  const layerOutputs =
      layerNames.map(layerName => model.getLayer(layerName).output);

  // Construct a mdoel that returns all the desired internal activations,
  // in addition to the final output of the original model.
  const compositeModel = tf.model(
      {inputs: model.input, outputs: layerOutputs.concat(model.outputs[0])});

  // `outputs` is an array of `tf.Tensor`s, including the internal activations
  // and the final output.
  const outputs = compositeModel.predict(inputImage);

  for (let i = 0; i < outputs.length - 1; ++i) {
    const layerName = layerNames[i];
    // Split the activation of the convolutional layer by filter.
    const activationTensors =
        tf.split(outputs[i], outputs[i].shape[outputs[i].shape.length - 1], -1);
    const actualFilters = filters <= activationTensors.length ?
        filters :
        activationTensors.length;
    const filePaths = [];
    let imageTensorShape;
    for (let j = 0; j < actualFilters; ++j) {
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
 * Calculate gradient-based class activation map and overlay it on input image.
 *
 * This function automatically finds the last convolutional layer, get its
 * output (activation) under the input image, weights its filters by the
 * gradient of the class output with respect to them, and then collapses along
 * the filter dimension.
 *
 * @param {tf.Sequential} model A TensorFlow.js sequential model, assumed to
 *   contain at least o
 * @param {number} classIndex Index to class in the model's final classification
 *   output.
 * @param {tf.Tensor4d} x Input image, assumed to have shape
 *   `[1, height, width, 3]`.
 * @param {number} overlayFactor Optional overlay factor.
 * @returns The input image with a heat-map representation of the class
 *   activation map overlaid on top of it, as float32-type `tf.Tensor4d` of
 *   shape `[1, height, width, 3]`.
 */
function gradClassActivationMap(model, classIndex, x, overlayFactor = 2.0) {
  // Try to locate the last conv layer of the model.
  let layerIndex = model.layers.length - 1;
  while (layerIndex >= 0) {
    if (model.layers[layerIndex].getClassName().startsWith('Conv')) {
      break;
    }
    layerIndex--;
  }
  tf.util.assert(
      layerIndex >= 0, `Failed to find a convolutional layer in model`);

  const lastConvLayer = model.layers[layerIndex];
  console.log(
      `Located last convolutional layer of the model at ` +
      `index ${layerIndex}: layer type = ${lastConvLayer.getClassName()}; ` +
      `layer name = ${lastConvLayer.name}`);

  // Get "sub-model 1", which goes from the original input to the output
  // of the last convolutional layer.
  const lastConvLayerOutput = lastConvLayer.output;
  const subModel1 =
      tf.model({inputs: model.inputs, outputs: lastConvLayerOutput});

  // Get "sub-model 2", which goes from the output of the last convolutional
  // layer to the original output.
  const newInput = tf.input({shape: lastConvLayerOutput.shape.slice(1)});
  layerIndex++;
  let y = newInput;
  while (layerIndex < model.layers.length) {
    y = model.layers[layerIndex++].apply(y);
  }
  const subModel2 = tf.model({inputs: newInput, outputs: y});

  return tf.tidy(() => {
    // This function runs sub-model 2 and extracts the slice of the probability
    // output that corresponds to the desired class.
    const convOutput2ClassOutput = (input) =>
        subModel2.apply(input, {training: true}).gather([classIndex], 1);
    // This is the gradient function of the output corresponding to the desired
    // class with respect to its input (i.e., the output of the last
    // convolutional layer of the original model).
    const gradFunction = tf.grad(convOutput2ClassOutput);

    // Calculate the values of the last conv layer's output.
    const lastConvLayerOutputValues = subModel1.apply(x);
    // Calculate the values of gradients of the class output w.r.t. the output
    // of the last convolutional layer.
    const gradValues = gradFunction(lastConvLayerOutputValues);

    // Pool the gradient values within each filter of the last convolutional
    // layer, resulting in a tensor of shape [numFilters].
    const pooledGradValues = tf.mean(gradValues, [0, 1, 2]);
    // Scale the convlutional layer's output by the pooled gradients, using
    // broadcasting.
    const scaledConvOutputValues =
        lastConvLayerOutputValues.mul(pooledGradValues);

    // Create heat map by averaging and collapsing over all filters.
    let heatMap = scaledConvOutputValues.mean(-1);

    // Normalize heatMap to the [0, 1] interval.
    heatMap = heatMap.relu();
    heatMap = heatMap.div(heatMap.max()).expandDims(-1);

    // Up-sample the heat map to the size of the input image.
    heatMap = tf.image.resizeBilinear(heatMap, [x.shape[1], x.shape[2]]);

    // Apply an RGB colormap on the heatMap.
    heatMap = utils.applyColorMap(heatMap);

    // To form the final output, overlay the color heat map on the input image.
    heatMap = heatMap.mul(overlayFactor).add(x.div(255));
    return heatMap.div(heatMap.max()).mul(255);
  });
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
    defaultValue: 80,
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
    const {modelOutput, layerName2FilePaths, layerName2ImageDims} =
        await writeInternalActivationAndGetOutput(
            model, layerNames, x, args.filters, args.outputDir);

    // Calculate internal activations and final output of the model.
    const topNum = 10;
    const {values: topKVals, indices: topKIndices} =
        tf.topk(modelOutput, topNum);
    const probScores = Array.from(await topKVals.data());
    const indices = Array.from(await topKIndices.data());
    const classNames =
        indices.map(index => imagenetClasses.IMAGENET_CLASSES[index]);

    console.log(`Top-${topNum} classes:`);
    for (let i = 0; i < topNum; ++i) {
      console.log(
          `  ${classNames[i]} (index=${indices[i]}): ` +
          `${probScores[i].toFixed(4)}`);
    }

    // Save the original input image and the top-10 classification results.
    const origImagePath = path.join(args.outputDir, path.basename(args.inputImage));
    shelljs.cp(args.inputImage, origImagePath);

    // Calculate Grad-CAM heatmap.
    const xWithCAMOverlay = gradClassActivationMap(model, indices[0], x);
    const camImagePath = path.join(args.outputDir, 'cam.png');
    await utils.writeImageTensorToFile(xWithCAMOverlay, camImagePath);
    console.log(`Written CAM-overlaid image to: ${camImagePath}`);    

    // Create manifest and write it to disk.
    const manifest = {
      indices,
      origImagePath,
      classNames,
      probScores,
      layerName2FilePaths,
      layerName2ImageDims,
      camImagePath,
      topIndex: indices[0],
      topProb: probScores[0],
      topClass: classNames[0]
    };
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
