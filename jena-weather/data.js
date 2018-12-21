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
  const month = +dateStrItems[1] - 1;  // month is 0-based in JS `Date` class.
  const year = +dateStrItems[2];

  const timeStrItems = items[1].split(':');
  const hours = +timeStrItems[0];
  const minutes = +timeStrItems[1];
  const seconds = +timeStrItems[2];

  return new Date(Date.UTC(year, month, day, hours, minutes, seconds));
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
    const csvLines = csvData.split('\n');

    // Parser header.
    const columnNames = csvLines[0].split(',');
    for (let i = 0; i < columnNames.length; ++i) {
      columnNames[i] = columnNames[i].slice(1, columnNames[i].length - 1);
    }

    this.dateTimeCol = columnNames.indexOf('Date Time');
    tf.util.assert(
        this.dateTimeCol === 0,
        `Unexpected date-time column index from ${JENA_WEATHER_CSV_PATH}`);

    this.dataColumnNames = columnNames.slice(1);
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
      const newDateTime = parseDateTime(items[0]);
      if (newDateTime == null) {
        console.log('Failed to parse date time:', items[0]);  // DEBUG
      }
      if (this.dateTime.length > 0 &&
          newDateTime.getTime() <=
              this.dateTime[this.dateTime.length - 1].getTime()) {
        console.warn(
            newDateTime, '<=', this.dateTime[this.dateTime.length - 1],
            items[0]);  // DEBUG
      }

      this.dateTime.push(newDateTime);
      this.data.push(items.slice(1).map(x => +x));
    }
    this.numRows = this.data.length;
    this.numColumns = this.data[0].length - 1;

    // TODO(cais): Normalization.
    await this.calculateMeansAndStddevs_();
  }

  /**
   * Calculate the means and standard deviations of every column.
   *
   * TensorFlow.js is used for acceleration.
   */
  async calculateMeansAndStddevs_() {
    tf.tidy(() => {
      // Instead of doing it on all columns at once, we do it
      // column by column, as doing it all at once causes WebGL OOM
      // on some machines.
      this.means = [];
      this.stddevs = [];
      for (const columnName of this.dataColumnNames) {
        // TODO(cais): See if we can relax this limit.
        const data =
            tf.tensor1d(this.getColumnData(columnName).slice(0, 6 * 24 * 365));
        const moments = tf.moments(data);
        this.means.push(moments.mean.dataSync());
        this.stddevs.push(Math.sqrt(moments.variance.dataSync()));
      }
      console.log('means:', this.means);
      console.log('stddevs:', this.stddevs);
    });
  }

  getDataColumnNames() {
    return this.dataColumnNames;
  }

  getTime(index) {
    return this.dateTime[index];
  }

  getColumnData(
      columnName, includeTime, normalize, beginIndex, length, stride) {
    const columnIndex = this.dataColumnNames.indexOf(columnName);
    tf.util.assert(columnIndex >= 0, `Invalid column name: ${columnName}`);

    if (beginIndex == null) {
      beginIndex = 0;
    }
    if (length == null) {
      length = this.numRows - beginIndex;
    }
    if (stride == null) {
      stride = 1;
    }
    const out = [];
    for (let i = beginIndex; i < beginIndex + length && i < this.numRows;
         i += stride) {
      let value = this.data[i][columnIndex];
      if (normalize) {
        value = (value - this.means[columnIndex]) / this.stddevs[columnIndex];
      }
      if (includeTime) {
        value = {x: this.dateTime[i].getTime(), y: value};
      }
      out.push(value);
    }
    return out;
  }

  /**
   * TODO(cais): Doc string.
   * @param {*} shuffle
   * @param {*} lookBack
   * @param {*} delay
   * @param {*} batchSize
   * @param {*} step
   * @param {*} minIndex
   * @param {*} maxIndex
   * @param {*} normalize
   */
  getIteratorFn(
      shuffle, lookBack, delay, batchSize, step, minIndex, maxIndex, normalize) {
    let i = minIndex + lookBack;
    // if (i + batchSize >= maxIndex) {  // TODO(cais): Check this.
    //   i = minIndex + lookBack;
    // }
    const lookBackSlices = Math.floor(lookBack / step);
    console.log(`lookBackSlices = ${lookBackSlices}`);  // DEBUG
    console.log(`this.tempCol = ${this.tempCol}`);      // DEBUG

    function iteratorFn() {
      const rows = [];
      if (shuffle) {
        const range = maxIndex - (minIndex + lookBack);
        for (let i = 0; i < batchSize; ++i) {
          const row = minIndex + lookBack + Math.floor(Math.random() * range);
          rows.push(row);
        }
      } else {
        for (let r = i; r < i + batchSize && r < maxIndex; ++r) {
          rows.push(r);
        }
      }

      const numExamples = rows.length;
      i += numExamples;

      const samples = tf.buffer([numExamples, lookBackSlices, this.numColumns]);
      const targets = tf.buffer([numExamples, 1]);
      for (let j = 0; j < numExamples; ++j) {
        const row = rows[j];
        let exampleRow = 0;
        for (let r = row - lookBack; r < row; r += step) {
          let exampleCol = 0;
          for (let n = 0; n < this.numColumns; ++n) {
            const columnIndex = n < this.tempCol - 1 ? n : n + 1;
            let value = this.data[r][columnIndex];
            if (normalize) {
              value =
                  (value - this.means[columnIndex]) / this.stddevs[columnIndex];
            }
            // DEBUG
            // console.log(
            //     `columnIndex=${columnIndex}, j=${j}, r=${exampleRow}, c=${exampleCol}: value=${value}`);
            samples.set(value, j, exampleRow, exampleCol++);
          }

          // TODO(cais): Remove the confusing this.tempCol - 1 thing.
          let value = this.data[r + delay][this.tempCol - 1];
          if (normalize) {
            value =
                (value - this.means[this.tempCol - 1]) / this.stddevs[this.tempCol - 1];
          }
          targets.set(value, j, 0);
          // TODO(cais): Make sure this doesn't go out of bound.
          exampleRow++;
        }
      }
      // TODO(cais): Memory management of samples and targets.
      return {
        value: [samples.toTensor(), targets.toTensor()],
        done: false
      };  // TODO(cais): Return done = true when done.
    }

    return iteratorFn.bind(this);
  }
}
