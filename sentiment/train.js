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


// import * as tfjsNpy from 'tfjs-npy';  // DEBUG
import * as tf from '@tensorflow/tfjs';
import {ArgumentParser} from 'argparse';

// import * as fs from 'fs';
import {loadData} from './data';  // DEBUG

/**
 * Create a model for IMDB sentiment analysis.
 *
 * @param {string} modelType Type of the model to be created.
 * @param {number} vocabularySize Input vocabulary size.
 * @param {number} embeddingSize Embedding vector size, used to
 *   configure the embedding layer.
 * @returns An uncompiled instance of `tf.Model`.
 */
function buildModel(modelType, vocabularySize, embeddingSize) {
  // TODO(cais): Bidirectional and dense-only.
  const model = tf.sequential();
  model.add(tf.layers.embedding(
      {inputDim: vocabularySize, outputDim: embeddingSize}));
  if (modelType === 'lstm') {
    const lstmUnits = 32;
    model.add(tf.layers.lstm({units: lstmUnits}));
  } else if (modelType === 'simpleRNN') {
    const simpleRNNUnits = 32;
    model.add(tf.layers.simpleRNN({units: simpleRNNUnits}));
  } else {
    throw new Error(`Unsupported model type: ${modelType}`);
  }
  model.add(tf.layers.dense({units: 1, activation: 'sigmoid'}));
  return model;
}

function parseArguments() {
  const parser = new ArgumentParser(
      {description: 'Train a model for IMDB sentiment analysis'});
  parser.addArgument('modelType', {
    type: 'string',
    optionStrings: ['lstm', 'simpleRNN'],
    help: 'Model type'
  });
  parser.addArgument('--numWords', {
    type: 'int',
    defaultValue: 10000,
    help: 'Number of words in the vocabulary'
  });
  parser.addArgument('--maxLen', {
    type: 'int',
    defaultValue: 100,
    help: 'Maximum sentence length in number of words. ' +
        'Shorter sentences will be padded; longers ones will be truncated.'
  });
  parser.addArgument(
      '--gpu', {action: 'storeTrue', help: 'Use GPU for training'});
  parser.addArgument('--optimizer', {
    type: 'string',
    defaultValue: 'adam',
    help: 'Optimizer to be used for model training'
  });
  parser.addArgument(
      '--epochs',
      {type: 'int', defaultValue: 5, help: 'Number of training epochs'});
  parser.addArgument(
      '--batchSize',
      {type: 'int', defaultValue: 128, help: 'Batch size for training'});
  parser.addArgument('--validationSplit', {
    type: 'float',
    defaultValue: 0.2,
    help: 'Validation split for training'
  });
  return parser.parseArgs();
}

async function main() {
  const args = parseArguments();

  if (args.gpu) {
    console.log('Using GPU for training');
    require('@tensorflow/tfjs-node-gpu');
  } else {
    console.log('Using CPU for training');
    require('@tensorflow/tfjs-node');
  }

  console.log('Loading data...');
  const {xTrain, yTrain, xTest, yTest} =
      loadData('./python/imdb', args.numWords, args.maxLen);
  // let xTrain = tfjsNpy.parse(toArrayBuffer(fs.readFileSync('/tmp/x_train.npy')));
  // let yTrain = tfjsNpy.parse(toArrayBuffer(fs.readFileSync('/tmp/y_train.npy')));
  // yTrain = yTrain.expandDims(1);
  // xTrain = tf.tensor2d(xTrain.dataSync(), xTrain.shape);
  // yTrain = tf.tensor2d(yTrain.dataSync(), yTrain.shape);
  // console.log(xTrain.shape);  // DEBUG
  // console.log(yTrain.shape);  // DEBUG

  console.log('Building model...');
  const embeddingSize = 32;
  const model = buildModel(args.modelType, args.numWords, embeddingSize);

  model.compile({
    loss: 'binaryCrossentropy',
    optimizer: args.optimizer,
    metrics: ['acc']
  });
  model.summary();

  console.log('Training model...');
  await model.fit(xTrain, yTrain, {
    epochs: args.epochs,
    batchSize: args.batchSize,
    validationSplit: args.validationSplit
  });

  console.log('Evaluating model...');
  const [testLoss, testAcc] =
      model.evaluate(xTest, yTest, {batchSize: args.batchSize});
  console.log(`Evaluation loss: ${(await testLoss.data())[0].toFixed(4)}`);
  console.log(`Evaluation accuracy: ${(await testAcc.data())[0].toFixed(4)}`);
}

function toArrayBuffer(buf) {
  var ab = new ArrayBuffer(buf.length);
  var view = new Uint8Array(ab);
  for (var i = 0; i < buf.length; ++i) {
      view[i] = buf[i];
  }
  return ab;
}

main();