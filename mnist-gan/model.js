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

function createGenerator(latentDim, channels) {
  // TODO(cais): Maybe use sequential.
  const gen = tf.sequential();

  gen.add(tf.layers.dense({units: 128 * 16 * 16, inputShape: [latentDim]}));
  gen.add(tf.layers.leakyReLU());
  gen.add(tf.layers.reshape({targetShape: [16, 16, 128]}));

  gen.add(tf.layers.conv2d({filters: 256, kernelSize: 5, padding: 'same'}));
  gen.add(tf.layers.leakyReLU());

  gen.add(tf.layers.conv2dTranspose(
      {filters: 256, kernelSize: 4, strides: 2, padding: 'same'}));
  gen.add(tf.layers.leakyReLU());

  gen.add(tf.layers.conv2d({filters: 256, kernelSize: 5, padding: 'same'}));
  gen.add(tf.layers.leakyReLU());

  gen.add(tf.layers.conv2d({filters: 256, kernelSize: 5, padding: 'same'}));
  gen.add(tf.layers.leakyReLU());

  gen.add(tf.layers.conv2d(
      {filters: channels, kernelSize: 7, activation: 'tanh', padding: 'same'}));
  return gen;
}

function createDiscriminator(height, width, channels) {
  // TODO(cais): Maybe use sequential.
  const disc = tf.sequential();

  disc.add(tf.layers.conv2d(
      {filters: 128, kernelSize: 3, inputShape: [height, width, channels]}));
  disc.add(tf.layers.leakyReLU());
  disc.add(tf.layers.conv2d({filters: 128, kernelSize: 4, strides: 2}));
  disc.add(tf.layers.leakyReLU());
  disc.add(tf.layers.conv2d({filters: 128, kernelSize: 4, strides: 2}));
  disc.add(tf.layers.leakyReLU());
  disc.add(tf.layers.conv2d({filters: 128, kernelSize: 4, strides: 2}));
  disc.add(tf.layers.leakyReLU());

  disc.add(tf.layers.flatten());
  disc.add(tf.layers.dropout({rate: 0.4}));
  disc.add(tf.layers.dense({units: 1, activation: 'sigmoid'}));

  // TODO(cais): clipValue?
  const optimizer = tf.train.rmsprop(0.0008, 1e-8);
  disc.compile({loss: 'binaryCrossentropy', optimizer});
  console.log('Discriminator model summary:');
  disc.summary();
  return disc;
}

function createGAN(generator, discriminator, latentDim) {
  discriminator.trainable = false;

  const ganInput = tf.input({shape: latentDim});
  const ganOutput = discriminator.apply(generator.apply(ganInput));
  const gan = tf.model({inputs: ganInput, outputs: ganOutput});

  // TODO(cais): clipValue?
  const optimizer = tf.train.rmsprop(0.0004, 1e-8);
  gan.compile({loss: 'binaryCrossentropy', optimizer});
  console.log('GAN model summary:');
  gan.summary();
  return gan;
}

module.exports = {
  createGenerator,
  createDiscriminator,
  createGAN
};
