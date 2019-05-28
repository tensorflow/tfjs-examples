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
 * Based on
 * https://github.com/fchollet/deep-learning-with-python-notebooks/blob/master/5.4-visualizing-what-convnets-learn.ipynb
 */

const argparse = require('argparse');
const fs = require('fs');
const path = require('path');
const shelljs = require('shelljs');
const tf = require('@tensorflow/tfjs');

const cam = require('./cam');
const imagenetClasses = require('./imagenet_classes');
const filters = require('./filters');
const utils = require('./utils');

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
    const imageTensor =
        filters.inputGradientAscent(model, layerName, i, iterations);
    const outputFilePath = path.join(outputDir, `${layerName}_${i + 1}.png`);
    filePaths.push(outputFilePath);
    await utils.writeImageTensorToFile(imageTensor, outputFilePath);
    imageTensor.dispose();
    console.log(`  --> ${outputFilePath}`);
  }
  return filePaths;
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
  const model = await tf.loadLayersModel(args.modelJsonUrl);
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
        await filters.writeInternalActivationAndGetOutput(
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
    const origImagePath =
        path.join(args.outputDir, path.basename(args.inputImage));
    shelljs.cp(args.inputImage, origImagePath);

    // Calculate Grad-CAM heatmap.
    const xWithCAMOverlay = cam.gradClassActivationMap(model, indices[0], x);
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
