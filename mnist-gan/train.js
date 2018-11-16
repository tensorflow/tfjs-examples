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

const tf = require('@tensorflow/tfjs');
require('@tensorflow/tfjs-node-gpu');
const argparse = require('argparse');

const data = require('./cifar10_data');
const model = require('./model');

const height = 32;
const width = 32;
const channels = 3;
const latentDim = 32;



async function run(iterations, batchSize, modelSavePath) {
  let {x: xTrain, y: yTrain} = await data.loadCifar10Data();
  yTrainData = await yTrain.data();
  console.log(yTrainData);
  const filteredExampleIndices = [];
  yTrainData.forEach((y, i) => {
    if (y === 6) {
      filteredExampleIndices.push(i);
    }
  });
  tf.util.shuffle(filteredExampleIndices);
  console.log(filteredExampleIndices.length);  // DEBUG

  xTrain = xTrain.gather(filteredExampleIndices).div(255);

  // TODO(cais): Select only frog images.
  console.log(xTrain.shape);  // DEBUG
  tf.dispose(yTrain);

  const generator = model.createGenerator(latentDim, channels);
  const discriminator = model.createDiscriminator(height, width, channels);

  const gan = model.createGAN(generator, discriminator, latentDim);

  let start = 0;
  for (let i = 0; i < iterations; ++i) {  // TODO(cais): Use iterations.
    let randomLatentVectors = tf.randomNormal([batchSize, latentDim]);
    const generatedImages = generator.predict(randomLatentVectors);
    // console.log('generate images shape:', generatedImages.shape);  // DEBUG

    // TODO(cais): Use real cifar10 images.
    if (start + batchSize >= xTrain.shape[0]) {
      start = 0;
    }
    const realImages = xTrain.slice(start, batchSize);
    start += batchSize;
    const combinedImages = tf.concat([generatedImages, realImages], 0);
    let labels =
        tf.concat([tf.ones([batchSize, 1]), tf.zeros([batchSize, 1])], 0);
    // TODO(cais): Use tidy!
    labels = labels.add(tf.randomUniform(labels.shape).mul(0.05));

    // console.log('combinedImges.shape:', combinedImages.shape);  // DEBUG
    // console.log('labels.shape:', labels.shape);          // DEBUG

    // discriminator.getWeights()[0].print();  // DEBUG
    const discHistory = await discriminator.fit(combinedImages, labels, {
      epochs: 1, batchSize, verbose: 0
    });
    const discLoss = discHistory.history.loss[0];

    // discriminator.getWeights()[0].print();  // DEBUG

    // TODO(cais): Memory clean up, by creating another const?
    randomLatentVectors = tf.randomNormal([batchSize, latentDim]);
    const misleadingTargets = tf.zeros([batchSize, 1]);
    const ganHistory = await gan.fit(randomLatentVectors, misleadingTargets, {
      epochs: 1, batchSize, verbose: 0
    });
    const aLoss = ganHistory.history.loss[0];
    console.log(
        `Iteration ${i}/${iterations}: ` +
        `disLoss=${discLoss}, ganLoss=${aLoss}`);
    tf.dispose([randomLatentVectors, generatedImages, realImages, combinedImages]);
  }
}

const parser = new argparse.ArgumentParser(
    {description: 'TensorFlow.js-Node MNIST Example.', addHelp: true});
parser.addArgument('--iterations', {
  type: 'int',
  defaultValue: 10000,
  help: 'Number of iteratios to train the GAN for.'
});
parser.addArgument('--batch_size', {
  type: 'int',
  defaultValue: 20,  // TODO(cais): Determine: DO NOT SUBMIT.
  help: 'Batch size to be used during model training.'
})
parser.addArgument('--model_save_path', {
  type: 'string',
  help: 'Path to which the model will be saved after training.'
});
const args = parser.parseArgs();

run(args.iterations, args.batch_size, args.model_save_path);
