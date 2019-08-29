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

const tf = require('@tensorflow/tfjs-node');
const argparse = require('argparse');
const https = require('https');
const fs = require('fs');
const createModel = require('./model');
const createDataset = require('./data');


const csvUrl =
    'https://storage.googleapis.com/tfjs-examples/abalone-node/abalone.csv';
const csvPath = './abalone.csv';

/**
 * Train a model with dataset, then save the model to a local folder.
 */
async function run(epochs, batchSize, savePath) {
  const datasetObj = await createDataset('file://' + csvPath);
  const model = createModel([datasetObj.numOfColumns]);
  // The dataset has 4177 rows. Split them into 2 groups, one for training and
  // one for validation. Take about 3500 rows as train dataset, and the rest as
  // validation dataset.
  const trainBatches = Math.floor(3500 / batchSize);
  const dataset = datasetObj.dataset.shuffle(1000).batch(batchSize);
  const trainDataset = dataset.take(trainBatches);
  const validationDataset = dataset.skip(trainBatches);

  await model.fitDataset(
      trainDataset, {epochs: epochs, validationData: validationDataset});

  await model.save(savePath);

  const loadedModel = await tf.loadLayersModel(savePath + '/model.json');
  const result = loadedModel.predict(
      tf.tensor2d([[0, 0.625, 0.495, 0.165, 1.262, 0.507, 0.318, 0.39]]));
  console.log(
      'The actual test abalone age is 10, the inference result from the model is ' +
      result.dataSync());
}

const parser = new argparse.ArgumentParser(
    {description: 'TensorFlow.js-Node Abalone Example.', addHelp: true});
parser.addArgument('--epochs', {
  type: 'int',
  defaultValue: 100,
  help: 'Number of epochs to train the model for.'
});
parser.addArgument('--batch_size', {
  type: 'int',
  defaultValue: 500,
  help: 'Batch size to be used during model training.'
})
parser.addArgument(
    '--savePath',
    {type: 'string', defaultValue: 'file://trainedModel', help: 'Path.'})
const args = parser.parseArgs();


const file = fs.createWriteStream(csvPath);
https.get(csvUrl, function(response) {
  response.pipe(file).on('close', async () => {
    run(args.epochs, args.batch_size, args.savePath);
  });
});
