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

 export async function plotData(container, xs, ys) {
   const xvals = await xs.data();
   const yvals = await ys.data();

   const values = Array.from(yvals).map((y, i) => {
     return {'x': xvals[i], 'y': yvals[i]};
   });

   const spec = {
     '$schema': 'https://vega.github.io/schema/vega-lite/v2.json',
     'width': 300,
     'height': 300,
     'data': {'values': values},
     'mark': 'point',
     'encoding': {
       'x': {'field': 'x', 'type': 'quantitative'},
       'y': {'field': 'y', 'type': 'quantitative'}
     }
   };

   return renderChart(container, spec, {actions: false});
 }

 export async function plotDataAndPredictions(container, xs, ys, preds) {
   const xvals = await xs.data();
   const yvals = await ys.data();
   const predVals = await preds.data();

   const values = Array.from(yvals).map((y, i) => {
     return {'x': xvals[i], 'y': yvals[i], pred: predVals[i]};
   });

   const spec = {
     '$schema': 'https://vega.github.io/schema/vega-lite/v2.json',
     'width': 300,
     'height': 300,
     'data': {'values': values},
     'layer': [
       {
         'mark': 'point',
         'encoding': {
           'x': {'field': 'x', 'type': 'quantitative'},
           'y': {'field': 'y', 'type': 'quantitative'}
         }
       },
       {
         'mark': 'line',
         'encoding': {
           'x': {'field': 'x', 'type': 'quantitative'},
           'y': {'field': 'pred', 'type': 'quantitative'},
           'color': {'value': 'tomato'}
         },
       }
     ]
   };

   return renderChart(container, spec, {actions: false});
 }

 export function renderCoefficients(container, coeff) {
   document.querySelector(container).innerHTML =
       `<span>a=${coeff.a.toFixed(3)}, b=${coeff.b.toFixed(3)}, c=${
           coeff.c.toFixed(3)},  d=${coeff.d.toFixed(3)}</span>`;
 }
