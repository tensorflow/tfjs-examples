/*
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

import * as tf from '@tensorflow/tfjs-node';

import {JenaWeatherData} from "./data";
import {buildGRUModel, buildMLPModel, buildSimpleRNNModel, getBaselineMeanAbsoluteError} from "./models";

describe('Model creation', () => {
  it('MLP', () => {
    const model = buildMLPModel([8, 9]);
    const config = model.layers[1].getConfig();
    expect(config.kernelRegularizer).toEqual(null);
    expect(model.inputs.length).toEqual(1);
    expect(model.inputs[0].shape).toEqual([null, 8, 9]);
    expect(model.outputs.length).toEqual(1);
    expect(model.outputs[0].shape).toEqual([null, 1]);
  });

  it('MLP with kernel regularizer', () => {
    const model = buildMLPModel([8, 9], tf.regularizers.l2({l2: 5e-2}));
    const config = model.layers[1].getConfig();
    expect(config.kernelRegularizer.config.l2).toEqual(5e-2);
    expect(model.inputs.length).toEqual(1);
    expect(model.inputs[0].shape).toEqual([null, 8, 9]);
    expect(model.outputs.length).toEqual(1);
    expect(model.outputs[0].shape).toEqual([null, 1]);
  });

  it('MLP with dropout', () => {
    const model = buildMLPModel([8, 9], null, 0.5);
    const denseConfig = model.layers[1].getConfig();
    expect(denseConfig.kernelRegularize).toEqual(undefined);
    const dropoutConfig = model.layers[model.layers.length - 2].getConfig();
    expect(dropoutConfig.rate).toEqual(0.5);
    expect(model.inputs.length).toEqual(1);
    expect(model.inputs[0].shape).toEqual([null, 8, 9]);
    expect(model.outputs.length).toEqual(1);
    expect(model.outputs[0].shape).toEqual([null, 1]);
  });
});

describe('RNN', () => {
  it('simpleRNN', () => {
    const model = buildSimpleRNNModel([8, 9]);
    expect(model.inputs.length).toEqual(1);
    expect(model.inputs[0].shape).toEqual([null, 8, 9]);
    expect(model.outputs.length).toEqual(1);
    expect(model.outputs[0].shape).toEqual([null, 1]);
  });

  it('buildGRUModel', () => {
    const model = buildGRUModel([8, 9]);
    expect(model.inputs.length).toEqual(1);
    expect(model.inputs[0].shape).toEqual([null, 8, 9]);
    expect(model.outputs.length).toEqual(1);
    expect(model.outputs[0].shape).toEqual([null, 1]);
  });
});

describe('getBaselineMeanAbsoluteError', () => {
  it('getBaselineMeanAbsoluteError', async () => {
    const dataset = new JenaWeatherData();
    await dataset.load();

    const baselineMAE = await getBaselineMeanAbsoluteError(
        dataset, true, false, 10 * 24 * 6, 6, 24 * 6);
    expect(baselineMAE).toBeCloseTo(0.29033);
  });
});
