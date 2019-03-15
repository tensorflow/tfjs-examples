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

/**
 * Train an Auxiliary Classifier Generative Adversarial Network (ACGAN) on the
 * MNIST dataset.
 *
 * To start the training:
 *
 * ```sh
 * yarn
 * yarn train
 * ```
 * 
 * If available, a CUDA GPU will give you a higher training speed:
 * 
 * ```sh
 * yarn
 * yarn train --gpu
 * ```
 *
 * To start the demo in the browser, do in a separate terminal:
 *
 * ```sh
 * yarn
 * yarn watch
 * ```
 *
 * It is recommended to use tfjs-node-gpu to train the model on a CUDA-enabled
 * GPU, as the convolution heavy operations run several times faster a GPU than
 * on the CPU with tfjs-node.
 *
 * For background of ACGAN, see:
 * - Augustus Odena, Christopher Olah, Jonathon Shlens. (2017) "Conditional
 *   image synthesis with auxiliary classifier GANs"
 *   https://arxiv.org/abs/1610.09585
 * 
 * The implementation is based on:
 *   https://github.com/keras-team/keras/blob/master/examples/mnist_acgan.py
 */

const fs = require('fs');
const path = require('path');

const argparse = require('argparse');
const data = require('./data');

// Number of classes in the MNIST dataset.
const NUM_CLASSES = 10;

// MNIST image size.
const IMAGE_SIZE = 28;

// The value of the tf object will be set dynamically, depending on whether
// the CPU (tfjs-node) or GPU (tfjs-node-gpu) backend is used. This is why
// `let` is used in lieu of the more conventional `const` here.
let tf = require('@tensorflow/tfjs');

/**
 * Build the generator part of ACGAN.
 *
 * The generator of ACGAN takes two inputs:
 *
 *   1. A random latent-space vector (the latent space is often referred to
 *      as "z-space" in GAN literature).
 *   2. A label for the desired image category (0, 1, ..., 9).
 *
 * It generates one output: the generated (i.e., fake) image.
 *
 * @param {number} latentSize Size of the latent space.
 * @returns {tf.LayersModel} The generator model.
 */
function buildGenerator(latentSize) {
  tf.util.assert(
      latentSize > 0 && Number.isInteger(latentSize),
      `Expected latent-space size to be a positive integer, but ` +
          `got ${latentSize}.`);

  const cnn = tf.sequential();

  // The number of units is chosen so that when the output is reshaped
  // and fed through the subsequent conv2dTranspose layers, the tensor
  // that comes out at the end has the exact shape that matches MNIST
  // images ([28, 28, 1]).
  cnn.add(tf.layers.dense(
      {units: 3 * 3 * 384, inputShape: [latentSize], activation: 'relu'}));
  cnn.add(tf.layers.reshape({targetShape: [3, 3, 384]}));

  // Upsample from [3, 3, ...] to [7, 7, ...].
  cnn.add(tf.layers.conv2dTranspose({
    filters: 192,
    kernelSize: 5,
    strides: 1,
    padding: 'valid',
    activation: 'relu',
    kernelInitializer: 'glorotNormal'
  }));
  cnn.add(tf.layers.batchNormalization());

  // Upsample to [14, 14, ...].
  cnn.add(tf.layers.conv2dTranspose({
    filters: 96,
    kernelSize: 5,
    strides: 2,
    padding: 'same',
    activation: 'relu',
    kernelInitializer: 'glorotNormal'
  }));
  cnn.add(tf.layers.batchNormalization());

  // Upsample to [28, 28, ...].
  cnn.add(tf.layers.conv2dTranspose({
    filters: 1,
    kernelSize: 5,
    strides: 2,
    padding: 'same',
    activation: 'tanh',
    kernelInitializer: 'glorotNormal'
  }));

  // Unlike most TensorFlow.js models, the generator part of an ACGAN has
  // two inputs:
  //   1. The latent vector that is used as the "seed" of the fake image
  //      generation.
  //   2. A class label that controls which of the ten MNIST digit classes
  //      the generated fake image is meant to belong to.

  // This is the z space commonly referred to in GAN papers.
  const latent = tf.input({shape: [latentSize]});

  // The desired label of the generated image, an integer in the interval
  // [0, NUM_CLASSES).
  const imageClass = tf.input({shape: [1]});

  // The desired label is converted to a vector of length `latentSize`
  // through embedding lookup.
  const classEmbedding = tf.layers.embedding({
    inputDim: NUM_CLASSES,
    outputDim: latentSize,
    embeddingsInitializer: 'glorotNormal'
  }).apply(imageClass);

  // Hadamard product between z-space and a class conditional embedding.
  const h = tf.layers.multiply().apply([latent, classEmbedding]);

  const fakeImage = cnn.apply(h);
  return tf.model({inputs: [latent, imageClass], outputs: fakeImage});
}

/**
 * Build the discriminator part of ACGAN.
 *
 * The discriminator model of ACGAN takes the input: an image of
 * MNIST format, of shape [batchSize, 28, 28, 1].
 *
 * It gives two outputs:
 *
 *   1. A sigmoid probability score between 0 and 1, for whether the
 *      discriminator judges the input image to be real (close to 1)
 *      or fake (closer to 0).
 *   2. Softmax probability scores for the 10 MNIST digit categories,
 *      which is the discriminator's 10-class classification result
 *      for the input image.
 *
 * @returns {tf.LayersModel} The discriminator model.
 */
function buildDiscriminator() {
  const cnn = tf.sequential();

  cnn.add(tf.layers.conv2d({
    filters: 32,
    kernelSize: 3,
    padding: 'same',
    strides: 2,
    inputShape: [IMAGE_SIZE, IMAGE_SIZE, 1]
  }));
  cnn.add(tf.layers.leakyReLU({alpha: 0.2}));
  cnn.add(tf.layers.dropout({rate: 0.3}));

  cnn.add(tf.layers.conv2d(
      {filters: 64, kernelSize: 3, padding: 'same', strides: 1}));
  cnn.add(tf.layers.leakyReLU({alpha: 0.2}));
  cnn.add(tf.layers.dropout({rate: 0.3}));

  cnn.add(tf.layers.conv2d(
      {filters: 128, kernelSize: 3, padding: 'same', strides: 2}));
  cnn.add(tf.layers.leakyReLU({alpha: 0.2}));
  cnn.add(tf.layers.dropout({rate: 0.3}));

  cnn.add(tf.layers.conv2d(
      {filters: 256, kernelSize: 3, padding: 'same', strides: 1}));
  cnn.add(tf.layers.leakyReLU({alpha: 0.2}));
  cnn.add(tf.layers.dropout({rate: 0.3}));

  cnn.add(tf.layers.flatten());

  const image = tf.input({shape: [IMAGE_SIZE, IMAGE_SIZE, 1]});
  const features = cnn.apply(image);

  // Unlike most TensorFlow.js models, the discriminator has two outputs.

  // The 1st output is the probability score assigned by the discriminator to
  // how likely the input example is a real MNIST image (as versus
  // a "fake" one generated by the generator).
  const realnessScore =
      tf.layers.dense({units: 1, activation: 'sigmoid'}).apply(features);
  // The 2nd output is the softmax probabilities assign by the discriminator
  // for the 10 MNIST digit classes (0 through 9). "aux" stands for "auxiliary"
  // (the namesake of ACGAN) and refers to the fact that unlike a standard GAN
  // (which performs just binary real/fake classification), the discriminator
  // part of ACGAN also performs multi-class classification.
  const aux = tf.layers.dense({units: NUM_CLASSES, activation: 'softmax'})
                  .apply(features);

  return tf.model({inputs: image, outputs: [realnessScore, aux]});
}

/**
 * Build a combined ACGAN model.
 *
 * @param {number} latentSize Size of the latent vector.
 * @param {tf.SymbolicTensor} imageClass Symbolic tensor for the desired image
 *   class. This is the other input to the generator.
 * @param {tf.LayersModel} generator The generator.
 * @param {tf.LayersModel} discriminator The discriminator.
 * @param {tf.Optimizer} optimizer The optimizer to be used for training the
 *   combined model.
 * @returns {tf.LayersModel} The combined ACGAN model, compiled.
 */
function buildCombinedModel(latentSize, generator, discriminator, optimizer) {
  // Latent vector. This is one of the two inputs to the generator.
  const latent = tf.input({shape: [latentSize]});
  // Desired image class. This is the second input to the generator.
  const imageClass = tf.input({shape: [1]});
  // Get the symbolic tensor for fake images generated by the generator.
  let fake = generator.apply([latent, imageClass]);
  let aux;

  // We only want to be able to train generation for the combined model.
  discriminator.trainable = false;
  [fake, aux] = discriminator.apply(fake);
  const combined =
      tf.model({inputs: [latent, imageClass], outputs: [fake, aux]});
  combined.compile({
    optimizer,
    loss: ['binaryCrossentropy', 'sparseCategoricalCrossentropy']
  });
  combined.summary();
  return combined;
}

// "Soft" one used for training the combined ACGAN model.
// This is an important trick in training GANs.
const SOFT_ONE = 0.95;

/**
 * Train the discriminator for one step.
 *
 * In this step, only the weights of the discriminator are updated. The
 * generator is not involved.
 *
 * The following steps are involved:
 *
 *   - Slice the training features and to get batch of real data.
 *   - Generate a random latent-space vector and a random label vector.
 *   - Feed the random latent-space vector and label vector to the
 *     generator and let it generate a batch of generated (i.e., fake) images.
 *   - Concatenate the real data and fake data; train the discriminator on
 *     the concatenated data for one step.
 *   - Obtain and return the loss values.
 *
 * @param {tf.Tensor} xTrain A tensor that contains the features of all the
 *   training examples.
 * @param {tf.Tensor} yTrain A tensor that contains the labels of all the
 *   training examples.
 * @param {number} batchStart Starting index of the batch.
 * @param {number} batchSize Size of the batch to draw from `xTrain` and
 *   `yTrain`.
 * @param {number} latentSize Size of the latent space (z-space).
 * @param {tf.LayersModel} generator The generator of the ACGAN.
 * @param {tf.LayersModel} discriminator The discriminator of the ACGAN.
 * @returns {number[]} The loss values from the one-step training as numbers.
 */
async function trainDiscriminatorOneStep(
    xTrain, yTrain, batchStart, batchSize, latentSize, generator,
    discriminator) {
  // TODO(cais): Remove tidy() once the current memory leak issue in tfjs-node
  //   and tfjs-node-gpu is fixed.
  const [x, y, auxY] = tf.tidy(() => {
    const imageBatch = xTrain.slice(batchStart, batchSize);
    const labelBatch = yTrain.slice(batchStart, batchSize).asType('float32');

    // Latent vectors.
    let zVectors = tf.randomUniform([batchSize, latentSize], -1, 1);
    let sampledLabels =
        tf.randomUniform([batchSize, 1], 0, NUM_CLASSES, 'int32')
            .asType('float32');

    const generatedImages =
        generator.predict([zVectors, sampledLabels], {batchSize: batchSize});

    const x = tf.concat([imageBatch, generatedImages], 0);

    const y = tf.tidy(
        () => tf.concat(
            [tf.ones([batchSize, 1]).mul(SOFT_ONE), tf.zeros([batchSize, 1])]));

    const auxY = tf.concat([labelBatch, sampledLabels], 0);
    return [x, y, auxY];
  });

  const losses = await discriminator.trainOnBatch(x, [y, auxY]);
  tf.dispose([x, y, auxY]);
  return losses;
}

/**
 * Train the combined ACGAN for one step.
 *
 * In this step, only the weights of the generator are updated.
 *
 * @param {number} batchSize Size of the fake-image batch to generate.
 * @param {number} latentSize Size of the latent space (z-space).
 * @param {tf.LayersModel} combined The instance of tf.LayersModel that combines
 *   the generator and the discriminator.
 * @returns {number[]} The loss values from the combined model as numbers.
 */
async function trainCombinedModelOneStep(batchSize, latentSize, combined) {
  // TODO(cais): Remove tidy() once the current memory leak issue in tfjs-node
  //   and tfjs-node-gpu is fixed.
  const [noise, sampledLabels, trick] = tf.tidy(() => {
    // Make new latent vectors.
    const zVectors = tf.randomUniform([batchSize, latentSize], -1, 1);
    const sampledLabels =
        tf.randomUniform([batchSize, 1], 0, NUM_CLASSES, 'int32')
            .asType('float32');

    // We want to train the generator to trick the discriminator.
    // For the generator, we want all the {fake, not-fake} labels to say
    // not-fake.
    const trick = tf.tidy(() => tf.ones([batchSize, 1]).mul(SOFT_ONE));
    return [zVectors, sampledLabels, trick];
  });

  const losses =
      combined.trainOnBatch([noise, sampledLabels], [trick, sampledLabels]);
  tf.dispose([noise, sampledLabels, trick]);
  return losses;
}

function parseArguments() {
  const parser = new argparse.ArgumentParser({
    description: 'TensorFlowj.js: MNIST ACGAN trainer example.',
    addHelp: true
  });
  parser.addArgument('--gpu', {
    action: 'storeTrue',
    help: 'Use tfjs-node-gpu for training (required CUDA GPU)'
  });
  parser.addArgument(
      '--epochs',
      {type: 'int', defaultValue: 100, help: 'Number of training epochs.'});
  parser.addArgument('--batchSize', {
    type: 'int',
    defaultValue: 100,
    help: 'Batch size to be used during training.'
  });
  parser.addArgument('--latentSize', {
    type: 'int',
    defaultValue: 100,
    help: 'Size of the latent space (z-space).'
  });
  parser.addArgument(
      '--learningRate',
      {type: 'float', defaultValue: 0.0002, help: 'Learning rate.'});
  parser.addArgument('--adamBeta1', {
    type: 'float',
    defaultValue: 0.5,
    help: 'Beta1 parameter of the ADAM optimizer.'
  });
  parser.addArgument('--generatorSavePath', {
    type: 'string',
    defaultValue: './dist/generator',
    help: 'Path to which the generator model will be saved after every epoch.'
  });
  parser.addArgument('--logDir', {
    type: 'string',
    help: 'Optional log directory to which the loss values will be written.'
  });
  return parser.parseArgs();
}

function makeMetadata(totalEpochs, currentEpoch, completed) {
  return {
    totalEpochs,
    currentEpoch,
    completed,
    lastUpdated: new Date().getTime()
  }
}

async function run() {
  const args = parseArguments();
  // Set the value of tf depending on whether the CPU or GPU version of
  // libtensorflow is used.
  if (args.gpu) {
    console.log('Using GPU');
    tf = require('@tensorflow/tfjs-node-gpu');
  } else {
    console.log('Using CPU');
    tf = require('@tensorflow/tfjs-node');
  }

  if (!fs.existsSync(path.dirname(args.generatorSavePath))) {
    fs.mkdirSync(path.dirname(args.generatorSavePath));
  }
  const saveURL = `file://${args.generatorSavePath}`;
  const metadataPath = path.join(args.generatorSavePath, 'acgan-metadata.json');

  // Build the discriminator.
  const discriminator = buildDiscriminator();
  discriminator.compile({
    optimizer: tf.train.adam(args.learningRate, args.adamBeta1),
    loss: ['binaryCrossentropy', 'sparseCategoricalCrossentropy']
  });
  discriminator.summary();

  // Build the generator.
  const generator = buildGenerator(args.latentSize);
  generator.summary();

  const optimizer = tf.train.adam(args.learningRate, args.adamBeta1);
  const combined = buildCombinedModel(
      args.latentSize, generator, discriminator, optimizer);

  await data.loadData();
  let {images: xTrain, labels: yTrain} = data.getTrainData();
  yTrain = tf.expandDims(yTrain.argMax(-1), -1);

  // Save the generator model once before starting the training.
  await generator.save(saveURL);

  let numTensors;
  let logWriter;
  if (args.logDir) {
    console.log(`Logging to tensorboard at logdir: ${args.logDir}`);
    logWriter = tf.node.summaryFileWriter(args.logDir);
  }

  let step = 0;
  for (let epoch = 0; epoch < args.epochs; ++epoch) {
    // Write some metadata to disk at the beginning of every epoch.
    fs.writeFileSync(
        metadataPath,
        JSON.stringify(makeMetadata(args.epochs, epoch, false)));

    const tBatchBegin = tf.util.now();

    const numBatches = Math.ceil(xTrain.shape[0] / args.batchSize);

    for (let batch = 0; batch < numBatches; ++batch) {
      const actualBatchSize = (batch + 1) * args.batchSize >= xTrain.shape[0] ?
          (xTrain.shape[0] - batch * args.batchSize) :
          args.batchSize;

      const dLoss = await trainDiscriminatorOneStep(
          xTrain, yTrain, batch * args.batchSize, actualBatchSize,
          args.latentSize, generator, discriminator);

      // Here we use 2 * actualBatchSize here, so that we have
      // the generator optimizer over an identical number of images
      // as the discriminator.
      const gLoss = await trainCombinedModelOneStep(
          2 * actualBatchSize, args.latentSize, combined);

      console.log(
          `epoch ${epoch + 1}/${args.epochs} batch ${batch + 1}/${
              numBatches}: ` +
          `dLoss = ${dLoss[0].toFixed(6)}, gLoss = ${gLoss[0].toFixed(6)}`);
      if (logWriter != null) {
        logWriter.scalar('dLoss', dLoss[0], step);
        logWriter.scalar('gLoss', gLoss[0], step);
        step++;
      }

      // Assert on no memory leak.
      // TODO(cais): Remove this check once the current memory leak in
      //   tfjs-node and tfjs-node-gpu is fixed.
      if (numTensors == null) {
        numTensors = tf.memory().numTensors;
      } else {
        tf.util.assert(
            tf.memory().numTensors === numTensors,
            `Leaked ${tf.memory().numTensors - numTensors} tensors`);
      }
    }

    await generator.save(saveURL);
    console.log(
        `epoch ${epoch + 1} elapsed time: ` +
        `${((tf.util.now() - tBatchBegin) / 1e3).toFixed(1)} s`);
    console.log(`Saved generator model to: ${saveURL}\n`);
  }

  // Write metadata to disk to indicate the end of the training.
  fs.writeFileSync(
      metadataPath,
      JSON.stringify(makeMetadata(args.epochs, args.epochs, true)));
}

if (require.main === module) {
  run();
}

module.exports = {
  buildCombinedModel,
  buildDiscriminator,
  buildGenerator,
  trainCombinedModelOneStep,
  trainDiscriminatorOneStep
};
