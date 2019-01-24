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

import * as tf from '@tensorflow/tfjs';
import {ArgumentParser} from 'argparse';
import * as fs from 'fs';
import * as path from 'path';
import * as shelljs from 'shelljs';

import {loadData, loadMetadataTemplate} from './data';

/**
 * Create a model for IMDB sentiment analysis.
 *
 * @param {string} modelType Type of the model to be created.
 * @param {number} vocabularySize Input vocabulary size.
 * @param {number} embeddingSize Embedding vector size, used to
 *   configure the embedding layer.
 * @returns An uncompiled instance of `tf.Model`.
 */
function buildModel(modelType, maxLen, vocabularySize, embeddingSize) {
  // TODO(cais): Bidirectional and dense-only.
  const model = tf.sequential();
  model.add(tf.layers.embedding({
    inputDim: vocabularySize,
    outputDim: embeddingSize,
    inputLength: maxLen
  }));
  if (modelType === 'flatten') {
    model.add(tf.layers.flatten());
  } else if (modelType === 'cnn') {
    model.add(tf.layers.dropout({rate: 0.2}));
    model.add(tf.layers.conv1d({
      filters: 250,
      kernelSize: 3,
      strides: 1,
      padding: 'valid',
      activation: 'relu'
    }));
    model.add(tf.layers.globalMaxPool1d({}));
    model.add(tf.layers.dense({units: 250, activation: 'relu'}));
  } else if (modelType === 'simpleRNN') {
    model.add(tf.layers.simpleRNN({units: 32}));
  } else if (modelType === 'lstm') {
    model.add(tf.layers.lstm({units: 32}));
  } else if (modelType === 'bidirectionalLSTM') {
    model.add(tf.layers.bidirectional(
        {layer: tf.layers.lstm({units: 32}), mergeMode: 'concat'}));
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
    optionStrings: ['flatten', 'cnn', 'simpleRNN', 'lstm', 'bidirectionalLSTM'],
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
  parser.addArgument('--embeddingSize', {
    type: 'int',
    defaultValue: 32,
    help: 'Number of word embedding dimensions'
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
  parser.addArgument('--modelSaveDir', {
    type: 'string',
    defaultValue: 'dist/resources',
    help: 'Optional path for model saving.'
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
      await loadData(args.numWords, args.maxLen);

  console.log('Building model...');
  const model = buildModel(
      args.modelType, args.maxLen, args.numWords, args.embeddingSize);

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

  // Save model.
  if (args.modelSaveDir != null && args.modelSaveDir.length > 0) {
    // Create base directory first.
    shelljs.mkdir('-p', args.modelSaveDir);

    // Load metadata template.
    console.log('Loading metadata template...');
    const metadata = await loadMetadataTemplate();

    // Save metadata.
    metadata.epochs = args.epochs;
    metadata.embedding_size = args.embeddingSize;
    metadata.max_len = args.maxLen;
    metadata.model_type = args.modelType;
    metadata.batch_size = args.batchSize;
    metadata.vocabulary_size = args.numWords;
    const metadataPath = path.join(args.modelSaveDir, 'metadata.json');
    fs.writeFileSync(metadataPath, JSON.stringify(metadata));
    console.log(`Saved metadata to ${metadataPath}`);

    // Save model artifacts.
    await model.save(`file://${args.modelSaveDir}`);
    console.log(`Saved model to ${args.modelSaveDir}`);
  }
}

main();
