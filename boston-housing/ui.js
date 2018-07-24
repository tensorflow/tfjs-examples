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

import renderChart from 'vega-embed';

const statusElement = document.getElementById('status');
export const updateStatus = (message) => {
  statusElement.value = message;
};

const losses = [];
export const plotData = async (epoch, trainLoss, valLoss) => {
  losses.push({
    'epoch': epoch,
    'loss': trainLoss,
    'split': 'Train Loss'
  });
  losses.push({
    'epoch': epoch,
    'loss': valLoss,
    'split': 'Validation Loss'
  });

  const spec = {
    '$schema': 'https://vega.github.io/schema/vega-lite/v2.json',
    'width': 300,
    'height': 300,
    'data': {'values': losses},
    'mark': 'line',
    'encoding': {
      'x': {'field': 'epoch', 'type': 'quantitative'},
      'y': {'field': 'loss', 'type': 'quantitative'},
      'color': {'field': 'split', 'type': 'nominal'}
    }
  };

  return renderChart('#plot', spec, {actions: false});
}
