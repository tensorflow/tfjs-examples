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

const path = require('path');
const _ = require('lodash');
const mkdirp = require('mkdirp');
const argparse = require('argparse');

// The tf module will be dynamically set depending on whether the `--gpu`
// flag is specified.
let tf;

const {
  DATASET_PATH,
  TRAIN_IMAGES_FILE,
  IMAGE_FLAT_SIZE,
  loadImages,
  previewImage,
  batchImages,
} = require('./data');

const {encoder, decoder, vae, vaeLoss} = require('./model');

let epochs;
let batchSize;

const INTERMEDIATE_DIM = 512;
const LATENT_DIM = 2;

/**
 * Train the auto encoder
 *
 * @param {number[][]} images Flattened images for VAE training.
 * @param {object} vaeOpts Options for the VAE model, including the following
 *   fields:
 *   - originaDim {number} Length of the input flattened image.
 *   - intermediateDim {number} Number of units of the intermediate (i.e.,
 *     hidden) dense layer.
 *   - latentDim {number} Dimensionality of the latent space (i.e,. z-space).
 * @param {string} savePath Path to which the decoder part of the VAE model
 *   will be saved after training.
 * @param {string?} logDir Optional path to log directory. If specified, the
 *   batch-by-batch loss values will be logged to the directory during training,
 *   so that the training process can be monitored using TensorBoard.
 */
async function train(images, vaeOpts, savePath, logDir) {
  const encoderModel = encoder(vaeOpts);
  const decoderModel = decoder(vaeOpts);
  const vaeModel = vae(encoderModel, decoderModel);

  let summaryWriter;
  if (logDir != null) {
    console.log(`Logging loss values to ${logDir}.`);
    console.log(
        `Use the following command to start the tensorboard backend server:`);
    console.log(`  tensorboard --logdir ${logDir}`);
    summaryWriter = tf.node.summaryFileWriter(logDir);
  }

  console.log('\n** Train Model **\n');

  // Because we use a custom loss function, we will use optimizer.minimize
  // instead of the more typical model.fit. We thus need to define an optimizer
  // and manage batching the data ourselves.

  // Cteate the optimizer
  const optimizer = tf.train.adam();

  // Group the data into batches.
  const batches = _.chunk(images, batchSize);

  // Run the train loop.
  let step = 0;
  for (let i = 0; i < epochs; i++) {
    console.log(`\nEpoch #${i + 1} of ${epochs}\n`);
    for (let j = 0; j < batches.length; j++) {
      const currentBatchSize = batches[j].length
      const batchedImages = batchImages(batches[j]);

      const reshaped =
          batchedImages.reshape([currentBatchSize, vaeOpts.originalDim]);

      // This is the model optimization step. We make a prediction
      // compute loss and return it so that optimizer.minimize can
      // adjust the weights of the model.
      optimizer.minimize(() => {
        const outputs = vaeModel.apply(reshaped);
        const loss = vaeLoss(reshaped, outputs, vaeOpts);
        process.stdout.write('.');
        if (j % 50 === 0) {
          console.log('\nLoss:', loss.dataSync()[0]);
        }
        if (summaryWriter != null) {
          summaryWriter.scalar('loss', loss, step++);
        }

        return loss;
      });
      tf.dispose([batchedImages, reshaped]);
    }
    console.log('');
    // Generate a preview after each epoch
    await generate(decoderModel, vaeOpts.latentDim);
  }

  console.log('done training');
  saveDecoder(savePath, decoderModel);
}

/**
 * Generate an image and preview it on the console.
 *
 * @param {tf.LayersModel} decoderModel Decoder portion of the VAE.
 * @param {number} latentDimSize Dimensionality of the latent space.
 */
async function generate(decoderModel, latentDimSize) {
  const targetZ = tf.zeros([latentDimSize]).expandDims();
  const generated = (decoderModel.predict(targetZ));

  await previewImage(generated.dataSync());
  tf.dispose([targetZ, generated]);
}

async function saveDecoder(savePath, decoderModel) {
  const decoderPath = path.join(savePath, 'decoder');
  mkdirp.sync(decoderPath);
  const saveURL = `file://${decoderPath}`;
  console.log(`Saving decoder to ${saveURL}`);
  await decoderModel.save(saveURL);
}

async function run(savePath, logDir) {
  // Load the data
  const dataPath = path.join(DATASET_PATH, TRAIN_IMAGES_FILE);
  const images = await loadImages(dataPath);
  console.log('Data Loaded', images.length);
  await previewImage(images[5]);
  await previewImage(images[50]);
  await previewImage(images[500]);
  // Start the training.
  const vaeOpts = {
    originalDim: IMAGE_FLAT_SIZE,
    intermediateDim: INTERMEDIATE_DIM,
    latentDim: LATENT_DIM
  };
  await train(images, vaeOpts, savePath, logDir);
}

(async function() {
  const parser = new argparse.ArgumentParser();
  parser.addArgument('--gpu', {
    action: 'storeTrue',
    help: 'Use tfjs-node-gpu for training (required CUDA and CuDNN)'
  });
  parser.addArgument('--epochs', {
    type: 'int',
    defaultValue: 100,
    help: 'Number of epochs to train the model for'
  });
  parser.addArgument('--batchSize', {
    type: 'int',
    defaultValue: 256,
    help: 'Batch size to be used during model training'
  });
  parser.addArgument('--logDir', {
    type: 'string',
    help: 'Directory to which the TensorBoard summaries will be saved ' +
    'during training.'
  });
  parser.addArgument('--savePath', {
    type: 'string',
    defaultValue: './models',
    help: 'Directory to which the decoder part of the VAE model will ' +
    'be saved after training. If the directory does not exist, it will be ' +
    'created.'
  });

  const args = parser.parseArgs();
  epochs = args.epochs;
  batchSize = args.batchSize;

  if (args.gpu) {
    console.log('Training using GPU.');
    tf = require('@tensorflow/tfjs-node-gpu');
  } else {
    console.log('Training using CPU.');
    tf = require('@tensorflow/tfjs-node');
  }

  await run(args.savePath, args.logDir);
})();
