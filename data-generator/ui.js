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

const generateSampleDataMessageElement =
    document.getElementById('generateSampleDataMessage');
const generatedDataContainerElement =
    document.getElementById('generatedDataContainer');
const batchSizeElement = document.getElementById('batchSizeInput');

/** Updates the message at the top of the sample data column. */
export function updateSampleDataMessage(message) {
  console.log(message);
  generateSampleDataMessageElement.textContent = message;
};

/** Returns current value of the batchSize a number. */
export function getBatchSize() {
  return batchSizeElement.valueAsNumber;
}

/**
 * Creates an HTML table, using div elements, to display the generated sample
 * data.
 * TODO(bileschi): describe the format of the input `generatedArray`
 */
export function updateSampleRowOutput(generatedArray) {
  generatedDataContainerElement.textContent = '';
  for (const sample of generatedArray) {
    const oneKeyRow = document.createElement('div');
    oneKeyRow.className = 'divTableRow';
    oneKeyRow.align = 'left';
    const playerOneDiv = document.createElement('div');
    const playerTwoDiv = document.createElement('div');
    const resultDiv = document.createElement('div');
    // TODO(bileschi): Style this better.
    playerOneDiv.className = 'divTableCell';
    playerTwoDiv.className = 'divTableCell';
    resultDiv.className = 'divTableCell';
    playerOneDiv.textContent = sample[0];
    playerTwoDiv.textContent = sample[1];
    resultDiv.textContent = sample[2];
    oneKeyRow.appendChild(playerOneDiv);
    oneKeyRow.appendChild(playerTwoDiv);
    oneKeyRow.appendChild(resultDiv);
    // add the div child to updateSampleRowOutput
    generatedDataContainerElement.appendChild(oneKeyRow);
  }
};
