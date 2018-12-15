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
 * Addition RNN example.
 *
 * Based on Python Keras example:
 *   https://github.com/keras-team/keras/blob/master/examples/addition_rnn.py
 */

import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';

import {JenaWeatherData} from './data';
import {logStatus} from './ui';

async function run() {
  logStatus('Loading Jena weather data...');
  const jenaWeatherData = new JenaWeatherData();
  await jenaWeatherData.load();
  logStatus('Done loading Jena weather data.');

  const dataColumnName = 'T (degC)';
  const dataChartContainer = document.getElementById('data-chart');
  const includeTime = true;
  // NOTE(cais): On a Linux workstation running latest Chrome, the length
  // limit seems to be around 120k.
  const values = [jenaWeatherData.getColumnData(dataColumnName, includeTime, 0, 7200)];
  tfvis.render.linechart(
      {values, series: [dataColumnName]}, dataChartContainer,
      {
        width: 800,
        height: 300,
        xLabel: 'Time',
        yLabel: dataColumnName,
      });
}

run();
