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

import {getDatasetStats, getNormalizedDatasets} from './data_housing';
import {compileModel} from './model_housing';

// tf will be imported dynamically depending on whether the flag `--gpu` is
// set.
let tf;

function parseArgs() {
  const parser = new argparse.ArgumentParser({
    description: 'TensorFlow.js Quantization Example: Training an MLP for the ' +
    'California Housing Price dataset.',
    addHelp: true
  });
  parser.addArgument('modelSavePath', {
    type: 'string',
    help: 'Path at which the model to be evaluated is saved.'
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

  const {count, featureMeans, featureStddevs, labelMean, labelStddev} =
      await getDatasetStats();

  const validationSplit = 0.2;
  const evaluationSplit = 0.1;
  const {evalXs, evalYs} =
      await getNormalizedDatasets(
          count, featureMeans, featureStddevs, labelMean, labelStddev,
          validationSplit, evaluationSplit);

  console.log(`Loading model from ${args.modelSavePath}...`);
  const model = await tf.loadLayersModel(`file://${args.modelSavePath}`);
  compileModel(model);

  console.log(`Performing evaluation...`);
  const t0 = tf.util.now();
  const evalOutput = model.evaluate(evalXs, evalYs);
  const t1 = tf.util.now();
  console.log(`\nEvaluation took ${(t1 - t0).toFixed(2)} ms.`);
  console.log(
      `\nEvaluation result:\n` +
      `  Loss = ${evalOutput.dataSync()[0].toFixed(6)}`);
}

if (require.main === module) {
  main();
}
