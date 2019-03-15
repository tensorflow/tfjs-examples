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
const gan = require('./gan');

describe('ACGAN', () => {
  it('buildGenerator', () => {
    const latentSize = 5;
    const generator = gan.buildGenerator(latentSize);
    expect(generator.inputs.length).toEqual(2);
    // Latent vector input.
    expect(generator.inputs[0].shape).toEqual([null, 5]);
    // MNIST digit class input.
    expect(generator.inputs[1].shape).toEqual([null, 1]);
    expect(generator.outputs.length).toEqual(1);
    // MNIST image tensor output.
    expect(generator.outputs[0].shape).toEqual([null, 28, 28, 1]);

    // Test generator.predict().
    const latentInput = tf.randomUniform([2, 5]);
    const classInput = tf.tensor2d([[0], [1]]);
    const numTensors0 = tf.memory().numTensors;
    const output = generator.predict([latentInput, classInput]);
    expect(output.shape).toEqual([2, 28, 28, 1]);
    tf.dispose(output);
    // Assert no memory leak.
    expect(tf.memory().numTensors).toEqual(numTensors0);
  });

  it('buildDiscriminator', () => {
    const discriminator = gan.buildDiscriminator();
    expect(discriminator.inputs.length).toEqual(1);
    // MNIST image input.
    expect(discriminator.inputs[0].shape).toEqual([null, 28, 28, 1]);
    expect(discriminator.outputs.length).toEqual(2);
    // Binary realness output.
    expect(discriminator.outputs[0].shape).toEqual([null, 1]);
    // 10-class classification output.
    expect(discriminator.outputs[1].shape).toEqual([null, 10]);
  });

  it('trainDiscriminatorOneStep', async () => {
    const numExamples = 4;
    const xTrain = tf.randomNormal([numExamples, 28, 28, 1]);
    const yTrain = tf.randomUniform([numExamples, 1]);
    let batchStart = 0;
    const batchSize = 2;
    const latentSize = 5;
    const generator = gan.buildGenerator(latentSize);
    const discriminator = gan.buildDiscriminator();
    discriminator.compile({
      optimizer: tf.train.adam(1e-3),
      loss: ['binaryCrossentropy', 'sparseCategoricalCrossentropy']
    });

    // Burn-in training call.
    await gan.trainDiscriminatorOneStep(
        xTrain, yTrain, batchStart, batchSize, latentSize, generator,
        discriminator);

    // Actually-tested training call.
    const numTensors0 = tf.memory().numTensors;
    batchStart += 2;
    const losses = await gan.trainDiscriminatorOneStep(
        xTrain, yTrain, batchStart, batchSize, latentSize, generator,
        discriminator);
    expect(losses.length).toEqual(3);
    // Total loss should be equal to the sum of the two component losses.
    expect(losses[0]).toBeCloseTo(losses[1] + losses[2]);
    expect(losses[1]).toBeGreaterThan(0);
    expect(losses[2]).toBeGreaterThan(0);
    // Assert no memory leak.
    expect(tf.memory().numTensors).toEqual(numTensors0);
  });

  it('trainCombinedModelOneStep', async () => {
    const latentSize = 5;
    const generator = gan.buildGenerator(latentSize);
    const discriminator = gan.buildDiscriminator();
    const optimizer = tf.train.adam(1e-3);
    const model = gan.buildCombinedModel(
        latentSize, generator, discriminator, optimizer);
    expect(model.inputs.length).toEqual(2);
    expect(model.inputs[0].shape).toEqual([null, 5]);
    expect(model.inputs[1].shape).toEqual([null, 1]);
    expect(model.outputs.length).toEqual(2);
    expect(model.outputs[0].shape).toEqual([null, 1]);
    expect(model.outputs[1].shape).toEqual([null, 10]);

    const batchSize = 4;

    // Burn-in training call.
    await gan.trainCombinedModelOneStep(batchSize, latentSize, model);

    const discriminatorOldWeights =
        discriminator.getWeights().map(w => w.dataSync());
    const generatorOldWeights =
        generator.getWeights().map(w => w.dataSync());

    // Actually-tested training call.
    const numTensors0 = tf.memory().numTensors;
    const losses =
        await gan.trainCombinedModelOneStep(batchSize, latentSize, model);
    expect(losses.length).toEqual(3);
    // Total loss should be equal to the sum of the two component losses.
    expect(losses[0]).toBeCloseTo(losses[1] + losses[2]);
    expect(losses[1]).toBeGreaterThan(0);
    expect(losses[2]).toBeGreaterThan(0);
    // Assert no memory leak.
    expect(tf.memory().numTensors).toEqual(numTensors0);

    const discriminatorNewWeights =
        discriminator.getWeights().map(w => w.dataSync());
    const generatorNewWeights =
        generator.getWeights().map(w => w.dataSync());
    // Assert that the discriminator's weights are not changed by the training
    // step.
    discriminatorOldWeights.forEach(((oldValue, i) => {
      const maxAbsDiff =
          tf.tensor1d(discriminatorNewWeights[i]).sub(tf.tensor1d(oldValue))
          .abs().max().arraySync();
      expect(maxAbsDiff).toEqual(0);
    }));
    // Assert that the generator's weights are changed by the training step.
    generatorNewWeights.forEach(((oldValue, i) => {
      const maxAbsDiff =
          tf.tensor1d(generatorOldWeights[i]).sub(tf.tensor1d(oldValue))
          .abs().max().arraySync();
      expect(maxAbsDiff).toBeGreaterThan(0);
    }));
  });
});
