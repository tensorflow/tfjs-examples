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
const createModel = require('./model');
const createDataset = require('./data');

const csvPath = 'file://./abalone.csv';

/**
 * Train a model with dataset, then save the model to a local folder.
 */
async function run(epochs, batchSize) {
  const datasetObj = await createDataset(csvPath);
  const model = createModel(datasetObj.numOfColumns);
  // The dataset has 4177 rows. Split them into 2 groups, one for training and
  // one for validation.
  const dataset = datasetObj.dataset.shuffle(1000).batch(batchSize);
  const trainDataset = dataset.take(8);
  const validationDataset = dataset.skip(8);

  await model.fitDataset(trainDataset, {
    epochs: epochs,
    validationData: validationDataset,
    callbacks: {
      onEpochEnd: async (epoch) => {
        console.log(`Epoch ${epoch + 1} of ${100} completed.`);
      }
    }
  });

  await model.save(`file://trainedModel`);
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
const args = parser.parseArgs();

run(args.epochs, args.batch_size);
