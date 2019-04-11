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
 * This file implements the code for a multilayer perceptron based variational
 * autoencoder and is a per of this code
 * https://github.com/keras-team/keras/blob/master/examples/variational_autoencoder.py
 *
 * See this tutorial for a description of how autoencoders work.
 * https://blog.keras.io/building-autoencoders-in-keras.html
 */

const tf = require('@tensorflow/tfjs');

/**
 * The encoder portion of the model.
 *
 * @param {object} opts encoder configuration, includnig the following fields:
 *   - originaDim {number} Length of the input flattened image.
 *   - intermediateDim {number} Number of units of the intermediate (i.e.,
 *     hidden) dense layer.
 *   - latentDim {number} Dimensionality of the latent space (i.e,. z-space).
 * @param {number} opts.originalDim number of dimensions in the original data.
 * @param {number} opts.intermediateDim number of dimensions in the bottleneck.
 * @param {number} opts.latentDim number of dimensions in latent space.
 * @returns {tf.LayersModel} the encoder model.
 */
function encoder(opts) {
  const {originalDim, intermediateDim, latentDim} = opts;

  const inputs = tf.input({shape: [originalDim], name: 'encoder_input'});
  const x = tf.layers.dense({units: intermediateDim, activation: 'relu'})
                .apply(inputs);
  const zMean = tf.layers.dense({units: latentDim, name: 'z_mean'}).apply(x);
  const zLogVar =
      tf.layers.dense({units: latentDim, name: 'z_log_var'}).apply(x);

  const z =
      new ZLayer({name: 'z', outputShape: [latentDim]}).apply([zMean, zLogVar]);

  const enc = tf.model({
    inputs: inputs,
    outputs: [zMean, zLogVar, z],
    name: 'encoder',
  });

  // console.log('Encoder Summary');
  // enc.summary();
  return enc;
}

/**
 * This layer implements the 'reparameterization trick' described in
 * https://blog.keras.io/building-autoencoders-in-keras.html.
 *
 * The implementation is in the call method.
 * Instead of sampling from Q(z|X):
 *    sample epsilon = N(0,I)
 *    z = z_mean + sqrt(var) * epsilon
 */
class ZLayer extends tf.layers.Layer {
  constructor(config) {
    super(config);
  }

  computeOutputShape(inputShape) {
    tf.util.assert(inputShape.length === 2 && Array.isArray(inputShape[0]),
        () => `Expected exactly 2 input shapes. But got: ${inputShape}`);
    return inputShape[0];
  }

  /**
   * The actual computation performed by an instance of ZLayer.
   *
   * @param {Tensor[]} inputs this layer takes two input tensors, z_mean and
   *     z_log_var
   * @return A tensor of the same shape as z_mean and z_log_var, equal to
   *     z_mean + sqrt(exp(z_log_var)) * epsilon, where epsilon is a random
   *     vector that follows the unit normal distribution (N(0, I)).
   */
  call(inputs, kwargs) {
    const [zMean, zLogVar] = inputs;
    const batch = zMean.shape[0];
    const dim = zMean.shape[1];

    const mean = 0;
    const std = 1.0;
    // sample epsilon = N(0, I)
    const epsilon = tf.randomNormal([batch, dim], mean, std);

    // z = z_mean + sqrt(var) * epsilon
    return zMean.add(zLogVar.mul(0.5).exp().mul(epsilon));
  }

  static get className() {
    return 'ZLayer';
  }
}
tf.serialization.registerClass(ZLayer);

/**
 * The decoder portion of the model.
 *
 * @param {*} opts decoder configuration
 * @param {number} opts.originalDim number of dimensions in the original data
 * @param {number} opts.intermediateDim number of dimensions in the bottleneck
 *                                      of the encoder
 * @param {number} opts.latentDim number of dimensions in latent space
 */
function decoder(opts) {
  const {originalDim, intermediateDim, latentDim} = opts;

  // The decoder model has a linear topology and hence could be constructed
  // with `tf.sequential()`. But we use the functional-model API (i.e.,
  // `tf.model()`) here nonetheless, for consistency with the encoder model
  // (see `encoder()` above).
  const input = tf.input({shape: [latentDim]});
  let y = tf.layers.dense({
    units: intermediateDim,
    activation: 'relu'
  }).apply(input);
  y = tf.layers.dense({
    units: originalDim,
    activation: 'sigmoid'
  }).apply(y);
  const dec = tf.model({inputs: input, outputs: y});

  // console.log('Decoder Summary');
  // dec.summary();
  return dec;
}

/**
 * The combined encoder-decoder pipeline.
 *
 * @param {tf.Model} encoder
 * @param {tf.Model} decoder
 *
 * @returns {tf.Model} the vae.
 */
function vae(encoder, decoder) {
  const inputs = encoder.inputs;
  const encoderOutputs = encoder.apply(inputs);
  const encoded = encoderOutputs[2];
  const decoderOutput = decoder.apply(encoded);
  const v = tf.model({
    inputs: inputs,
    outputs: [decoderOutput, ...encoderOutputs],
    name: 'vae_mlp',
  })

  // console.log('VAE Summary');
  // v.summary();
  return v;
}

/**
 * The custom loss function for VAE.
 *
 * @param {tf.tensor} inputs the encoder inputs a batched image tensor
 * @param {[tf.tensor]} outputs the vae outputs, [decoderOutput,
 *     ...encoderOutputs]
 * @param {number} vaeOpts.originalDim number of dimensions in the original data
 */
function vaeLoss(inputs, outputs) {
  return tf.tidy(() => {
    const originalDim = inputs.shape[1];
    const decoderOutput = outputs[0];
    const zMean = outputs[1];
    const zLogVar = outputs[2];

    // First we compute a 'reconstruction loss' terms. The goal of minimizing
    // tihs term is to make the model outputs match the input data.
    const reconstructionLoss =
        tf.losses.meanSquaredError(inputs, decoderOutput).mul(originalDim);

    // binaryCrossEntropy can be used as an alternative loss function
    // const reconstructionLoss =
    //  tf.metrics.binaryCrossentropy(inputs, decoderOutput).mul(originalDim);

    // Next we compute the KL-divergence between zLogVar and zMean, minimizing
    // this term aims to make the distribution of latent variable more normally
    // distributed around the center of the latent space.
    let klLoss = zLogVar.add(1).sub(zMean.square()).sub(zLogVar.exp());
    klLoss = klLoss.sum(-1).mul(-0.5);

    return reconstructionLoss.add(klLoss).mean();
  });
}

module.exports = {
  vae,
  encoder,
  decoder,
  vaeLoss,
}
