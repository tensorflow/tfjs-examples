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

// TODO(bileschi): Is this the right way to get the type of the object
// returned from tfjs-data?
import { TensorContainer } from "@tensorflow/tfjs-core/dist/tensor_types";

const statusElement = document.getElementById('status') as HTMLTextAreaElement;
const rowCountOutputElement = document.getElementById('rowCountOutput');
const columnNamesOutputElement = document.getElementById('columnNamesOutput');
const sampleRowMessageElement = document.getElementById('sampleRowMessage');
const sampleRowOutputContainerElement = document.getElementById('sampleRowOutputContainer');
const whichSampleInputElement = document.getElementById('whichSampleInput') as HTMLInputElement;

export const updateStatus = (message: string) => {
  console.log(message);
  statusElement.value = message;
};

export const updateRowCountOutput = (message: string) => {
  console.log(message);
  rowCountOutputElement.textContent = message;
};

export const updateColumnNamesOutput = (message: string) => {
  console.log(message);
  columnNamesOutputElement.textContent = message;
};

export const updateSampleRowMessage = (message: string) => {
  console.log(message);
  sampleRowMessageElement.textContent = message;
};

export const updateSampleRowOutput = (rawRow: any) => {
  sampleRowOutputContainerElement.textContent = "";
  const row = rawRow as object;
  for (const key in row) {
    if (row.hasOwnProperty(key)) {
      console.log(key);
      sampleRowOutputContainerElement.textContent += key + " ";
    }
  }
};

export const getSampleIndex = () => {
  return whichSampleInputElement.valueAsNumber;
}

export const getQueryElement = () =>
  document.getElementById('queryURL') as HTMLTextAreaElement;

