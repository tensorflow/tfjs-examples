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

import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-node';

import {TextData} from './data';
import {createModel, compileModel, fitModel, generateText, sample} from './model';

// tslint:disable:max-line-length
const FAKE_TEXT = `Lorem ipsum dolor sit amet, consectetur adipiscing elit. Suspendisse tempor aliquet justo non varius. Curabitur eget convallis velit. Vivamus malesuada, tortor ut finibus posuere, libero lacus eleifend felis, sit amet tempus dolor magna id nibh. Praesent non turpis libero. Praesent luctus, neque vitae suscipit suscipit, arcu neque aliquam justo, eget gravida diam augue nec lorem. Etiam scelerisque vel nibh sit amet maximus. Praesent et dui quis elit bibendum elementum a eget velit. Mauris porta lorem ac porttitor congue. Vestibulum lobortis ultrices velit, vitae condimentum elit ultrices a. Vivamus rutrum ultrices eros ac finibus. Orci varius natoque penatibus et magnis dis parturient montes, nascetur ridiculus mus. Morbi a purus a nibh eleifend convallis. Praesent non turpis volutpat, imperdiet lacus in, cursus tellus. Etiam elit velit, ornare sit amet nulla vel, aliquam iaculis mauris.

Phasellus sed sem ut justo sollicitudin cursus at sed neque. Proin tempor finibus nisl, nec aliquam leo porta at. Nullam vel mauris et neque pellentesque laoreet sit amet eu risus. Sed sed ante sed enim hendrerit commodo. Etiam blandit aliquet molestie. Nullam dictum imperdiet enim, quis scelerisque nunc ultricies sit amet. Praesent dictum dictum lobortis. Sed ut ipsum at orci commodo congue.

Aenean pharetra mollis erat, id convallis ante elementum at. Cras semper turpis nec lorem tempus ultrices. Sed eget purus vel est blandit dictum. Praesent auctor, sapien non consequat pellentesque, risus orci sagittis leo, at cursus nibh nisi vel quam. Morbi et orci id quam dictum efficitur ac iaculis nisl. Donec at nunc et nibh accumsan malesuada eu in odio. Donec quis elementum turpis. Vestibulum pretium rhoncus orci, nec gravida nisl hendrerit pellentesque. Cras imperdiet odio a quam mollis, in aliquet neque efficitur. Praesent at tincidunt ipsum. Maecenas neque risus, pretium ut orci sit amet, dignissim auctor dui. Sed finibus nunc elit, rhoncus ornare dui pharetra vitae. Sed ut iaculis ex. Quisque quis molestie ligula. Vivamus egestas rhoncus mollis.

Pellentesque volutpat ipsum vitae ex interdum, eu rhoncus dolor fringilla. Suspendisse potenti. Maecenas in sem leo. Curabitur vestibulum porta vulputate. Nunc quis consectetur enim. Aliquam congue, augue in commodo porttitor, sem tellus posuere augue, ut aliquam sapien massa in est. Duis convallis pellentesque vehicula. Mauris ipsum urna, congue consequat posuere sed, euismod nec mauris. Praesent sollicitudin scelerisque scelerisque. Ut commodo nisl vitae nunc feugiat auctor. Praesent imperdiet magna facilisis nunc vulputate, vel suscipit leo consequat. Duis fermentum rutrum ipsum a laoreet. Nunc dictum libero in quam pellentesque, sit amet tempus tellus suscipit. Curabitur pharetra erat bibendum malesuada rhoncus.

Donec laoreet leo ligula, ut condimentum mi placerat ut. Sed pretium sollicitudin nisl quis tincidunt. Proin id nisl ornare, interdum lorem quis, posuere lacus. Cras cursus mollis scelerisque. Mauris mattis mi sed orci feugiat, et blandit velit tincidunt. Donec ultrices leo vel tellus tincidunt, id vehicula mi commodo. Nulla egestas mollis massa. Etiam blandit nisl eu risus luctus viverra. Mauris eget mi sem.

`;
// tslint:enable:max-line-length

describe('text-generation model', () => {
  function createTextDataForTest(sampleLen, sampleStep = 1) {
    return new TextData('LoremIpsum', FAKE_TEXT, sampleLen, sampleStep);
  }

  it('createModel: 1 LSTM layer', () => {
    const model = createModel(20, 52, 32);
    expect(model.layers.length).toEqual(2);
    expect(model.inputs.length).toEqual(1);
    expect(model.inputs[0].shape).toEqual([null, 20, 52]);
    expect(model.outputs.length).toEqual(1);
    expect(model.outputs[0].shape).toEqual([null, 52]);
  });

  it('createModel: 2 LSTM layers', () => {
    const model = createModel(20, 52, [32, 16]);
    expect(model.layers.length).toEqual(3);
    expect(model.inputs.length).toEqual(1);
    expect(model.inputs[0].shape).toEqual([null, 20, 52]);
    expect(model.outputs.length).toEqual(1);
    expect(model.outputs[0].shape).toEqual([null, 52]);
  });

  it('compileModel', () => {
    const model = createModel(20, 52, 32);
    compileModel(model, 1e-2);
    expect(model.optimizer != null).toEqual(true);
  });

  it('fitModel', async () => {
    const sampleLen = 10;
    const textData = createTextDataForTest(sampleLen);
    const model = createModel(textData.sampleLen(), textData.charSetSize(), 16);
    compileModel(model, 1e-2);

    const epochs = 2;
    const examplesPerEpoch = 16;
    const batchSize = 4;
    const validationSplit = 0.25;
    const batchEndBatches = [];
    const epochEndEpochs = [];
    const callback = {
      onBatchEnd: async (batch, logs) =>  {
        batchEndBatches.push(batch);
      },
      onEpochEnd: async (epoch, logs) => {
        epochEndEpochs.push(epoch);
      }
    }
    await fitModel(
        model, textData, epochs, examplesPerEpoch, batchSize, validationSplit,
        callback);
    expect(batchEndBatches).toEqual([0, 1, 2, 0, 1, 2]);
    expect(epochEndEpochs).toEqual([0, 0]);
  });

  it('generateText', async () => {
    const sampleLen = 10;
    const textData = createTextDataForTest(sampleLen);
    const model = createModel(textData.sampleLen(), textData.charSetSize(), 16);

    const sentenceIndices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
    const text = await generateText(model, textData, sentenceIndices, 12, 0.5);
    // Assert that the original indices are not altered.
    expect(sentenceIndices).toEqual([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    expect(typeof text).toEqual('string');
    expect(text.length).toEqual(12);
  });

  it('sample: temperature = 0', async () => {
    const sampleLen = 10;
    const textData = createTextDataForTest(sampleLen);
    const charSetSize = textData.charSetSize();

    const probsBuffer = tf.buffer([charSetSize]);
    probsBuffer.set(1, charSetSize - 2);
    const probs = probsBuffer.toTensor();

    const temperature = 0;
    const sampled = sample(probs, temperature);

    // Sampling under temperature === 0 should be deterministic.
    expect(sampled).toEqual(charSetSize - 2);
  });

  it('sample: temperature = 0.75', async () => {
    const sampleLen = 10;
    const textData = createTextDataForTest(sampleLen);
    const charSetSize = textData.charSetSize();

    let probs = tf.randomUniform([charSetSize]);
    probs = probs.div(probs.sum());

    const temperature = 0.75;
    const uniqueSamples = [];
    for (let i = 0; i < 16; ++i) {
      const sampled = sample(probs, temperature);
      expect(sampled).toBeGreaterThanOrEqual(0);
      expect(sampled).toBeLessThan(charSetSize);
      expect(Number.isInteger(sampled)).toEqual(true);
      if (uniqueSamples.indexOf(sampled) === -1) {
        uniqueSamples.push(sampled);
      }
    }

    // Sampling under temperature 0.75 should be random.
    expect(uniqueSamples.length).toBeGreaterThan(1);
  });
});
