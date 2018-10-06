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
require('@tensorflow/tfjs-node');
const argparse = require('argparse');

const data = require('./data');
const model = require('./model');

const height = 32;
const width = 32;
const channels = 3;
const latentDim = 32;

async function run(iterations, batchSize, modelSavePath) {
  const generator = model.createGenerator(latentDim, channels);
  generator.summary();  // DEBUG

  const discriminator = model.createDiscriminator(height, width, channels);
  discriminator.summary();  // DEBUG

  const gan = model.createGAN(generator, discriminator, latentDim);
  gan.summary();  // DEBUG

  let start = 0;
  for (let i = 0; i < 1; ++i) {  // TODO(cais): Use iterations.
    let randomLatentVectors = tf.randomNormal([batchSize, latentDim]);
    console.log(randomLatentVectors.shape);  // DEBUG
    const generatedImages = generator.predict(randomLatentVectors);

    const stop = start + batchSize;
    // TODO(cais): Use real cifar10 images.
    const realImages = tf.randomNormal([batchSize, height, width, channels]);
    const combinedImages = tf.concat([generatedImages, realImages], 0);
    const labels =
        tf.concat([tf.ones([batchSize, 1]), tf.zeros([batchSize, 1])], 0);
    labels.add(tf.randomNormal(labels.shape).mul(0.05));

    console.log(combinedImages.shape);  // DEBUG
    console.log(labels.shape);          // DEBUG

    // discriminator.getWeights()[0].print();  // DEBUG
    const discHistory = await discriminator.fit(combinedImages, labels, {
      epochs: 1, batchSize, verbose: 0
    });
    const discLoss = discHistory.history.loss[0];
    console.log(discLoss);  // DEBUG
    // discriminator.getWeights()[0].print();  // DEBUG

    randomLatentVectors = tf.randomNormal([batchSize, latentDim]);
    const misleadingTargets = tf.zeros([batchSize, 1]);
    const ganHistory = await gan.fit(randomLatentVectors, misleadingTargets, {
      epochs: 1, batchSize, verbose: 0
    });
    const ganLoss = ganHistory.ganHistory.loss[0];
    console.log(ganLoss);  // DEBUG
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
  defaultValue: 20,
  help: 'Batch size to be used during model training.'
})
parser.addArgument('--model_save_path', {
  type: 'string',
  help: 'Path to which the model will be saved after training.'
});
const args = parser.parseArgs();

run(args.iterations, args.batch_size, args.model_save_path);
