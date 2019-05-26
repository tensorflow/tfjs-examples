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

import {FashionMnistDataset, MnistDataset} from './data_mnist';
import {compileModel} from './model_mnist';

// The `tf` module will be loaded dynamically depending on whether
// `--gpu` is specified in the command-line flags.
let tf;

function parseArgs() {
  const parser = new argparse.ArgumentParser({
    description:
        'TensorFlow.js Quantization Example: Evaluating an MNIST Model',
    addHelp: true
  });
  parser.addArgument('dataset', {
    type: 'string',
    help: 'Name of the dataset ({mnist, fashion-mnist}).'
  });
  parser.addArgument('modelSavePath', {
    type: 'string',
    help: 'Path at which the model to be evaluated is saved.'
  });
  parser.addArgument('--batchSize', {
    type: 'int',
    defaultValue: 128,
    help: 'Batch size to be used during model training.'
  });
  parser.addArgument('--gpu', {
    action: 'storeTrue',
    help: 'Use tfjs-node-gpu for evaluation (requires CUDA-enabled ' +
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
  if (args.dataset === 'fashion-mnist') {
    dataset = new FashionMnistDataset();
  } else if (args.dataset === 'mnist') {
    dataset = new MnistDataset();
  } else {
    throw new Error(`Unrecognized dataset name: ${args.dataset}`);
  }
  await dataset.loadData();
  const {images: testImages, labels: testLabels} = dataset.getTestData();

  console.log(`Loading model from ${args.modelSavePath}...`);
  const model = await tf.loadLayersModel(`file://${args.modelSavePath}`);
  compileModel(model);

  console.log(`Performing evaluation...`);
  const t0 = tf.util.now();
  const evalOutput = model.evaluate(testImages, testLabels);
  const t1 = tf.util.now();
  console.log(`\nEvaluation took ${(t1 - t0).toFixed(2)} ms.`);
  console.log(
      `\nEvaluation result:\n` +
      `  Loss = ${evalOutput[0].dataSync()[0].toFixed(6)}; `+
      `Accuracy = ${evalOutput[1].dataSync()[0].toFixed(6)}`);
}

if (require.main === module) {
  main();
}
