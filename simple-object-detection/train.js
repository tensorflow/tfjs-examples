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

const fs = require('fs');
const path = require('path');

const argparse = require('argparse');
const canvas = require('canvas');
const tf = require('@tensorflow/tfjs');
const synthesizer = require('./synthetic_images');

const CANVAS_SIZE = 224;  // Matches the input size of MobileNet.

// Name prefixes of layers that will be unfrozen during fine-tuning.
const topLayerGroupNames = ['conv_pw_9', 'conv_pw_10', 'conv_pw_11'];

// Name of the layer that will become the top layer of the truncated base.
const topLayerName =
    `${topLayerGroupNames[topLayerGroupNames.length - 1]}_relu`;

// Used to scale the first column (0-1 shape indicator) of `yTrue`
// in order to ensure balanced contributions to the final loss value
// from shape and bounding-box predictions.
const LABEL_MULTIPLIER = [CANVAS_SIZE, 1, 1, 1, 1];

/**
 * Custom loss function for object detection.
 *
 * The loss function is a sum of two losses
 * - shape-class loss, computed as binaryCrossentropy and scaled by
 *   `classLossMultiplier` to match the scale of the bounding-box loss
 *   approximatey.
 * - bounding-box loss, computed as the meanSquaredError between the
 *   true and predicted bounding boxes.
 * @param {tf.Tensor} yTrue True labels. Shape: [batchSize, 5].
 *   The first column is a 0-1 indicator for whether the shape is a triangle
 *   (0) or a rectangle (1). The remaining for columns are the bounding boxes
 *   for the target shape: [left, right, top, bottom], in unit of pixels.
 *   The bounding box values are in the range [0, CANVAS_SIZE).
 * @param {tf.Tensor} yPred Predicted labels. Shape: the same as `yTrue`.
 * @return {tf.Tensor} Loss scalar.
 */
function customLossFunction(yTrue, yPred) {
  return tf.tidy(() => {
    // Scale the the first column (0-1 shape indicator) of `yTrue` in order
    // to ensure balanced contributions to the final loss value
    // from shape and bounding-box predictions.
    return tf.metrics.meanSquaredError(yTrue.mul(LABEL_MULTIPLIER), yPred);
  });
}

/**
 * Loads MobileNet, removes the top part, and freeze all the layers.
 *
 * The top removal and layer freezing are preparation for transfer learning.
 *
 * Also gets handles to the layers that will be unfrozen during the fine-tuning
 * phase of the training.
 *
 * @return {tf.Model} The truncated MobileNet, with all layers frozen.
 */
async function loadTruncatedBase() {
  // TODO(cais): Add unit test.
  const mobilenet = await tf.loadLayersModel(
      'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json');

  // Return a model that outputs an internal activation.
  const fineTuningLayers = [];
  const layer = mobilenet.getLayer(topLayerName);
  const truncatedBase =
      tf.model({inputs: mobilenet.inputs, outputs: layer.output});
  // Freeze the model's layers.
  for (const layer of truncatedBase.layers) {
    layer.trainable = false;
    for (const groupName of topLayerGroupNames) {
      if (layer.name.indexOf(groupName) === 0) {
        fineTuningLayers.push(layer);
        break;
      }
    }
  }

  tf.util.assert(
      fineTuningLayers.length > 1,
      `Did not find any layers that match the prefixes ${topLayerGroupNames}`);
  return {truncatedBase, fineTuningLayers};
}

/**
 * Build a new head (i.e., output sub-model) that will be connected to
 * the top of the truncated base for object detection.
 *
 * @param {tf.Shape} inputShape Input shape of the new model.
 * @returns {tf.Model} The new head model.
 */
function buildNewHead(inputShape) {
  const newHead = tf.sequential();
  newHead.add(tf.layers.flatten({inputShape}));
  newHead.add(tf.layers.dense({units: 200, activation: 'relu'}));
  // Five output units:
  //   - The first is a shape indictor: predicts whether the target
  //     shape is a triangle or a rectangle.
  //   - The remaining four units are for bounding-box prediction:
  //     [left, right, top, bottom] in the unit of pixels.
  newHead.add(tf.layers.dense({units: 5}));
  return newHead;
}

/**
 * Builds object-detection model from MobileNet.
 *
 * @returns {[tf.Model, tf.layers.Layer[]]}
 *   1. The newly-built model for simple object detection.
 *   2. The layers that can be unfrozen during fine-tuning.
 */
async function buildObjectDetectionModel() {
  const {truncatedBase, fineTuningLayers} = await loadTruncatedBase();

  // Build the new head model.
  const newHead = buildNewHead(truncatedBase.outputs[0].shape.slice(1));
  const newOutput = newHead.apply(truncatedBase.outputs[0]);
  const model = tf.model({inputs: truncatedBase.inputs, outputs: newOutput});

  return {model, fineTuningLayers};
}

(async function main() {
  // Data-related settings.
  const numCircles = 10;
  const numLines = 10;

  const parser = new argparse.ArgumentParser();
  parser.addArgument('--gpu', {
    action: 'storeTrue',
    help: 'Use tfjs-node-gpu for training (required CUDA and CuDNN)'
  });
  parser.addArgument(
      '--numExamples',
      {type: 'int', defaultValue: 2000, help: 'Number of training exapmles'});
  parser.addArgument('--validationSplit', {
    type: 'float',
    defaultValue: 0.15,
    help: 'Validation split to be used during training'
  });
  parser.addArgument('--batchSize', {
    type: 'int',
    defaultValue: 128,
    help: 'Batch size to be used during training'
  });
  parser.addArgument('--initialTransferEpochs', {
    type: 'int',
    defaultValue: 100,
    help: 'Number of training epochs in the initial transfer ' +
        'learning (i.e., 1st) phase'
  });
  parser.addArgument('--fineTuningEpochs', {
    type: 'int',
    defaultValue: 100,
    help: 'Number of training epochs in the fine-tuning (i.e., 2nd) phase'
  });
  const args = parser.parseArgs();

  if (args.gpu) {
    console.log('Training using GPU.');
    require('@tensorflow/tfjs-node-gpu');
  } else {
    console.log('Training using CPU.');
    require('@tensorflow/tfjs-node');
  }

  const modelSaveURL = 'file://./dist/object_detection_model';

  const tBegin = tf.util.now();
  console.log(`Generating ${args.numExamples} training examples...`);
  const synthDataCanvas = canvas.createCanvas(CANVAS_SIZE, CANVAS_SIZE);
  const synth =
      new synthesizer.ObjectDetectionImageSynthesizer(synthDataCanvas, tf);
  const {images, targets} =
      await synth.generateExampleBatch(args.numExamples, numCircles, numLines);

  const {model, fineTuningLayers} = await buildObjectDetectionModel();
  model.compile({loss: customLossFunction, optimizer: tf.train.rmsprop(5e-3)});
  model.summary();

  // Initial phase of transfer learning.
  console.log('Phase 1 of 2: initial transfer learning');
  await model.fit(images, targets, {
    epochs: args.initialTransferEpochs,
    batchSize: args.batchSize,
    validationSplit: args.validationSplit
  });

  // Fine-tuning phase of transfer learning.
  // Unfreeze layers for fine-tuning.
  for (const layer of fineTuningLayers) {
    layer.trainable = true;
  }
  model.compile({loss: customLossFunction, optimizer: tf.train.rmsprop(2e-3)});
  model.summary();

  // Do fine-tuning.
  // The batch size is reduced to avoid CPU/GPU OOM. This has
  // to do with the unfreezing of the fine-tuning layers above,
  // which leads to higher memory consumption during backpropagation.
  console.log('Phase 2 of 2: fine-tuning phase');
  await model.fit(images, targets, {
    epochs: args.fineTuningEpochs,
    batchSize: args.batchSize / 2,
    validationSplit: args.validationSplit
  });

  // Save model.
  // First make sure that the base directory dists.
  const modelSavePath = modelSaveURL.replace('file://', '');
  const dirName = path.dirname(modelSavePath);
  if (!fs.existsSync(dirName)) {
    fs.mkdirSync(dirName);
  }
  await model.save(modelSaveURL);
  console.log(`Model training took ${(tf.util.now() - tBegin) / 1e3} s`);
  console.log(`Trained model is saved to ${modelSaveURL}`);
  console.log(
      `\nNext, run the following command to test the model in the browser:`);
  console.log(`\n  yarn watch`);
})();
