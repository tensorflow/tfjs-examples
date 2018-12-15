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
 * Data object for Jena Weather data.
 * 
 * TODO(cais): Say something about the origin of the data.
 */

import * as tf from '@tensorflow/tfjs';

const JENA_WEATHER_CSV_PATH =
    'https://storage.googleapis.com/learnjs-data/jena_climate/jena_climate_2009_2016.csv';

function parseDateTime(str) {
  const items = str.split(' ');
  const dateStr = items[0];
  const dateStrItems = dateStr.split('.');
  const day = +dateStrItems[0];
  const month = +dateStrItems[1];
  const year = +dateStrItems[2];

  const timeStrItems = items[1].split(':');
  const hours = +timeStrItems[0];
  const minutes = +timeStrItems[1];
  const seconds = +timeStrItems[2];

  return new Date(year, month, day, hours, minutes, seconds);
}

/**
 * A class that fetches the sprited MNIST dataset and provide data as
 * tf.Tensors.
 */
export class JenaWeatherData {
  constructor() {}

  async load() {
    const csvData = await (await fetch(JENA_WEATHER_CSV_PATH)).text();
    
    // Parse CSV file.
    console.log(csvData.length);  // DEBUG
    const csvLines = csvData.split('\n');
    console.log(csvLines.length);  // DEBUG

    // Parser header.
    const columnNames = csvLines[0].split(',');
    for (let i = 0; i < columnNames.length; ++i) {
      columnNames[i] = columnNames[i].slice(1, columnNames[i].length - 1);
    }

    console.log(columnNames);  // DEBUG
    this.dateTimeCol = columnNames.indexOf('Date Time');
    tf.util.assert(
        this.dateTimeCol === 0, 
        `Unexpected date-time column index from ${JENA_WEATHER_CSV_PATH}`);

    this.dataColumnNames = columnNames.slice(1);
    console.log(this.dataColumnNames);  // DEBUG
    this.tempCol = columnNames.indexOf('T (degC)');
    tf.util.assert(
        this.tempCol >= 1, 
        `Unexpected T (degC) column index from ${JENA_WEATHER_CSV_PATH}`);

    this.dateTime = [];
    this.data = [];
    for (let i = 1; i < csvLines.length; ++i) {
      const line = csvLines[i].trim();
      if (line.length === 0) {
        continue;
      }
      const items = line.split(',');
      this.dateTime.push(parseDateTime(items[0]));
      this.data.push(items.slice(1).map(x => +x));
    }
    this.numRows = this.dateTime.length;
    console.log(this.numRows);
    console.log(this.data.length);
    console.log(this.data[200]);

    // TODO(cais): Normalization.
  }

  getDataColumnNames() {
    return this.dataColumnNames;
  }

  getColumnData(columnName, includeTime, beginIndex, endIndex, stride) {
    // TODO(cais): Handle includeDateTime.
    const columnIndex = this.dataColumnNames.indexOf(columnName);
    tf.util.assert(columnIndex >= 0, `Invalid column name: ${columnName}`);

    if (beginIndex == null) {
      beginIndex = 0;
    }
    if (endIndex == null) {
      endIndex = this.numRows;
    }
    if (stride == null) {
      stride = 1;
    }
    const out = [];
    for (let i = beginIndex; i < endIndex; i += stride) {
      let value = this.data[i][columnIndex];  
      if (includeTime) {
        value = {
          x: this.dateTime[i].getTime(),
          y: value
        };
      }
      out.push(value);
    }
    return out;
  }
}
