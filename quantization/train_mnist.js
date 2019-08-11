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

import * as argparse from 'argparse';
import * as fs from 'fs';
import * as path from 'path';
import * as shelljs from 'shelljs';

// tf will be imported dynamically depending on whether the flag `--gpu` is
// set.
let tf;

import {FashionMnistDataset, MnistDataset} from './data_mnist';
import {createModel as createMnistModel} from './model_mnist';
import {createModel as createFashionMnistModel} from './model_fashion_mnist';

function parseArgs() {
  const parser = new argparse.ArgumentParser({
    description: 'TensorFlow.js Quantization Example: Training an MNIST Model',
    addHelp: true
  });
  parser.addArgument('dataset', {
    type: 'string',
    help: 'Name of the dataset ({mnist, fashion-mnist}).'
  });
  parser.addArgument('--epochs', {
    type: 'int',
    defaultValue: 100,
    help: 'Number of epochs to train the model for.'
  });
  parser.addArgument('--batchSize', {
    type: 'int',
    defaultValue: 128,
    help: 'Batch size to be used during model training.'
  });
  parser.addArgument('--validationSplit', {
    type: 'float',
    defaultValue: 0.15,
    help: 'Validation split used for training.'
  });
  parser.addArgument('--modelSavePath', {
    type: 'string',
    defaultValue: './models/',
    help: 'Path to which the model will be saved after training.'
  });
  parser.addArgument('--gpu', {
    action: 'storeTrue',
    help: 'Use tfjs-node-gpu for training (requires CUDA-enabled ' +
    'GPU and supporting drivers and libraries.'
  });
  return parser.parseArgs();
}

async function main() {
  const args = parseArgs();
  if (args.gpu) {
    tf = require('@tensorflow/tfjs-node-gpu');
  } else {
    tf = require('@tensorflow/tfjs-node');
  }

  let dataset;
  let model;
  if (args.dataset === 'fashion-mnist') {
    dataset = new FashionMnistDataset();
    model = createFashionMnistModel();
  } else if (args.dataset === 'mnist') {
    dataset = new MnistDataset();
    model = createMnistModel();
  } else {
    throw new Error(`Unrecognized dataset name: ${args.dataset}`);
  }
  await dataset.loadData();
  const {images: trainImages, labels: trainLabels} = dataset.getTrainData();

  model.summary();

  await model.fit(trainImages, trainLabels, {
    epochs: args.epochs,
    batchSize: args.batchSize,
    validationSplit: args.validationSplit,
    callbacks: tf.callbacks.earlyStopping({patience: 20})
  });

  const {images: testImages, labels: testLabels} = dataset.getTestData();
  const evalOutput = model.evaluate(testImages, testLabels);

  console.log(
      `\nEvaluation result:\n` +
      `  Loss = ${evalOutput[0].dataSync()[0].toFixed(6)}; `+
      `Accuracy = ${evalOutput[1].dataSync()[0].toFixed(6)}`);

  const modelSavePath = path.join(args.modelSavePath, args.dataset, 'original');
  if (modelSavePath != null) {
    if (!fs.existsSync(path.dirname(modelSavePath))) {
      shelljs.mkdir('-p', path.dirname(modelSavePath));
    }
    await model.save(`file://${modelSavePath}`);
    console.log(`Saved model to path: ${modelSavePath}`);
  }
}

if (require.main === module) {
  main();
}
