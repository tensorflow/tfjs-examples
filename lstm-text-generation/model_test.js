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

import {createModel, compileModel} from './model';
import '@tensorflow/tfjs-node';

describe('text-generation model', () => {
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

  // TODO(cais): Add unit test for fitModel();
});