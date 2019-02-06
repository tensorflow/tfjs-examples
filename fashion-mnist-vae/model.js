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


/**
 * Thils file implements the code for a multilayer perceptron based variational
 * autoencoder and is a per of this code
 * https://github.com/keras-team/keras/blob/master/examples/variational_autoencoder.py
 *
 * See this tutorial for a description of how autoencoders work.
 * https://blog.keras.io/building-autoencoders-in-keras.html
 */


/**
 * The encoder portion of the model
 * @param {*} opts
 */
function encoder(opts) {
  const {originalDim, intermediateDim, latentDim} = opts;

  const inputs = tf.input({shape: [originalDim], name: 'encoder_input'});
  const x = tf.layers.dense({units: intermediateDim, activation: 'relu'})
                .apply(inputs);
  const zMean = tf.layers.dense({units: latentDim, name: 'z_mean'}).apply(x);
  const zLogVar =
      tf.layers.dense({units: latentDim, name: 'z_log_var'}).apply(x);
  const z = new zLayer({name: 'z'}, [latentDim]).apply([zMean, zLogVar]);

  const enc = tf.model({
    inputs: inputs,
    outputs: [zMean, zLogVar, z],
    name: 'encoder',
  })

  // console.log('Encoder Summary')
  // enc.summary();
  return enc;
}

class zLayer extends tf.layers.Layer {
  constructor(config, outputShape) {
    super(config);
    this._outputShape = outputShape;
  }

  computeOutputShape(inputShape) {
    return this._outputShape;
  }

  call(inputs, kwargs) {
    const [z_mean, z_log_var] = inputs;
    const batch = z_mean.shape[0];
    const dim = z_mean.shape[1];
    // #by default, random_normal has mean = 0 and std = 1.0
    const mean = 0;
    const std = 1.0;
    const epsilon = tf.randomNormal([batch, dim], mean, std);
    return z_mean.add(z_log_var.mul(0.5).exp()).mul(epsilon);
  }

  getClassName() {
    return 'zLayer';
  }
}


/**
 * The decoder portion of the model
 * @param {*} opts
 */
function decoder(opts) {
  const {originalDim, intermediateDim, latentDim} = opts;

  const latentInputs = tf.input({shape: [latentDim], name: 'z_sampling'});
  const x = tf.layers.dense({units: intermediateDim, activation: 'relu'})
                .apply(latentInputs);
  const outputs =
      tf.layers.dense({units: originalDim, activation: 'sigmoid'}).apply(x);

  const dec = tf.model({
    inputs: latentInputs,
    outputs: outputs,
    name: 'decoder',
  });

  // console.log('Decoder Summary')
  // dec.summary();
  return dec;
}


/**
 * The combined encoder-decorder pipeline.
 * @param {*} opts
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
  // console.log('VAE Summary')
  // v.summary();
  return v;
}

/**
 * The custom loss function for VAE
 * @param {*} inputs
 * @param {*} outputs
 * @param {*} vaeOpts
 */
function vaeLoss(inputs, outputs, vaeOpts) {
  const {originalDim} = vaeOpts;
  const decoderOutput = outputs[0];
  const zMean = outputs[1];
  const zLogVar = outputs[2];

  const reconstructionLoss =
      tf.losses.meanSquaredError(inputs, decoderOutput).mul(originalDim);

  let klLoss = zLogVar.add(1).sub(zMean.square()).sub(zLogVar.exp());
  klLoss = klLoss.sum(-1);
  klLoss = klLoss.mul(-0.5)

  return reconstructionLoss.add(klLoss).mean();
}

module.exports = {
  vae,
  encoder,
  decoder,
  vaeLoss,
}
