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
 * Train recurrent neural networks (RNNs) for temperature prediction.
 *
 * This script drives the RNN training process in the Node.js environment
 * using tfjs-node or tfjs-node-gpu (see the `--gpu` flag).
 *
 * - See [data.js](./data.js) for how the Jena weather dataset is loaded.
 * - See [models.js](./train.js) for the detailed model creation and training
 *   logic.
 */

import {ArgumentParser} from 'argparse';

import {JenaWeatherData} from './data';
import {buildModel, getBaselineMeanAbsoluteError, trainModel} from './models';

global.fetch = require('node-fetch');

function parseArguments() {
  const parser =
      new ArgumentParser({description: 'Train RNNs for Jena weather problem'});
  parser.addArgument('--modelType', {
    type: 'string',
    defaultValue: 'gru',
    optionStrings: ['baseline', 'gru', 'simpleRNN'],
    // TODO(cais): Add more model types, e.g., gru with recurrent dropout.
    help: 'Type of the model to train. Use "baseline" to compute the ' +
    'commonsense baseline prediction error.'
  });
  parser.addArgument('--gpu', {
    action: 'storeTrue',
    help: 'Use GPU'
  });
  parser.addArgument('--lookBack', {
    type: 'int',
    defaultValue: 10 * 24 * 6,
    help: 'Look-back period (# of rows) for generating features'
  });
  parser.addArgument('--step', {
    type: 'int',
    defaultValue: 6,
    help: 'Step size (# of rows) used for generating features'
  });
  parser.addArgument('--delay', {
    type: 'int',
    defaultValue: 24 * 6,
    help: 'How many steps (# of rows) in the future to predict the ' +
        'temperature for'
  });
  parser.addArgument('--normalize', {
    defaultValue: true,
    help: 'Used normalized feature values (default: true)'
  });
  parser.addArgument('--includeDateTime', {
    action: 'storeTrue',
    help: 'Used date and time features (default: false)'
  });
  parser.addArgument(
      '--batchSize',
      {type: 'int', defaultValue: 128, help: 'Batch size for training'});
  parser.addArgument(
      '--epochs',
      {type: 'int', defaultValue: 20, help: 'Number of training epochs'});
  parser.addArgument( '--earlyStoppingPatience', {
    type: 'int',
    defaultValue: 2,
    help: 'Optional patience number for EarlyStoppingCallback'
   });
  parser.addArgument('--logDir', {
    type: 'string',
    help: 'Optional tensorboard log directory, to which the loss and ' +
    'accuracy will be logged during model training.'
  });
  parser.addArgument('--logUpdateFreq', {
    type: 'string',
    defaultValue: 'batch',
    optionStrings: ['batch', 'epoch'],
    help: 'Frequency at which the loss and accuracy will be logged to ' +
    'tensorboard.'
  });
  return parser.parseArgs();
}

async function main() {
  const args = parseArguments();
  let tfn;
  if (args.gpu) {
    console.log('Using GPU for training.');
    tfn = require('@tensorflow/tfjs-node-gpu');
  } else {
    console.log('Using CPU for training.');
    tfn = require('@tensorflow/tfjs-node');
  }

  const jenaWeatherData = new JenaWeatherData();
  console.log(`Loading Jena weather data...`);
  await jenaWeatherData.load();

  if (args.modelType === 'baseline') {
    console.log('Calculating commonsense baseline mean absolute error...');
    const baselineError = await getBaselineMeanAbsoluteError(
        jenaWeatherData, args.normalize, args.includeDateTime, args.lookBack,
        args.step, args.delay);
    console.log(
        `Commonsense baseline mean absolute error: ` +
        `${baselineError.toFixed(6)}`);
  } else {
    let numFeatures = jenaWeatherData.getDataColumnNames().length;
    const model = buildModel(
        args.modelType, Math.floor(args.lookBack / args.step), numFeatures);

    let callback = [];
    if (args.logDir != null) {
      console.log(
          `Logging to tensorboard. ` +
          `Use the command below to bring up tensorboard server:\n` +
          `  tensorboard --logdir ${args.logDir}`);
      callback.push(tfn.node.tensorBoard(args.logDir, {
        updateFreq: args.logUpdateFreq
      }));
    }
    if (args.earlyStoppingPatience != null) {
      console.log(
          `Using earlyStoppingCallback with patience ` +
          `${args.earlyStoppingPatience}.`);
      callback.push(tfn.callbacks.earlyStopping({
        patience: args.earlyStoppingPatience
      }));
    }

    await trainModel(
        model, jenaWeatherData, args.normalize, args.includeDateTime,
        args.lookBack, args.step, args.delay, args.batchSize, args.epochs,
        callback);
  }
}

if (require.main === module) {
  main();
}
