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

import * as tfvis from '@tensorflow/tfjs-vis';

const statusElement = document.getElementById('status');
export function updateStatus(message) {
  statusElement.innerText = message;
};

export async function plotLosses(trainLogs) {
  return tfvis.show.history(
      document.getElementById('plotLoss'), trainLogs, ['loss', 'val_loss'], {
        width: 450,
        height: 320,
        xLabel: 'Epoch',
        yLabel: 'Loss',
      });
}

export async function plotAccuracies(trainLogs) {
  tfvis.show.history(
      document.getElementById('plotAccuracy'), trainLogs, ['acc', 'val_acc'], {
        width: 450,
        height: 320,
        xLabel: 'Epoch',
        yLabel: 'Accuracy',
      });
}

const rocValues = [];
const rocSeries = [];

/**
 * Plot a ROC curve.
 * @param {number[]} fprs False positive rates.
 * @param {number[]} tprs True positive rates, must have the same length as
 *   `fprs`.
 * @param {number} epoch Epoch number.
 */
export async function plotROC(fprs, tprs, epoch) {
  epoch++;  // Convert zero-based to one-based.

  // Store the series name in the list of series
  const seriesName = 'epoch ' +
      (epoch < 10 ? `00${epoch}` : (epoch < 100 ? `0${epoch}` : `${epoch}`))
  rocSeries.push(seriesName);

  const newSeries = [];
  for (let i = 0; i < fprs.length; i++) {
    newSeries.push({
      x: fprs[i],
      y: tprs[i],
    });
  }
  rocValues.push(newSeries);

  return tfvis.render.linechart(
      {values: rocValues, series: rocSeries},
      document.getElementById('rocCurve'),
      {
        width: 450,
        height: 320,
      },
  );
}
