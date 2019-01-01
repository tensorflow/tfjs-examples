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
 * The data used in this demo is the
 * [Jena weather archive
 * dataset](https://www.kaggle.com/pankrzysiu/weather-archive-jena).
 */

import * as tf from '@tensorflow/tfjs';

const FAST_JENA_WEATHER_CSV_PATH = './jena_climate_2009_2016.csv';
const JENA_WEATHER_CSV_PATH =
    'https://storage.googleapis.com/learnjs-data/jena_climate/jena_climate_2009_2016.csv';

/**
 * Parse the date-time string from the Jena weather CSV file.
 *
 * @param {*} str The date time string with a format that looks like:
 *   "17.01.2009 22:10:00"
 * @returns date: A JavaScript Date object.
 *          normalizedDayOfYear: Day of the year, normalized between 0 and 1.
 *          normalizedTimeOfDay: Time of the day, normalized between 0 and 1.
 */
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

  const date = new Date(Date.UTC(year, month, day, hours, minutes, seconds));
  const yearOnset = new Date(year, 0, 1);
  const normalizedDayOfYear = (date - yearOnset) / (366 * 1000 * 60 * 60 * 24);
  const dayOnset = new Date(year, month, day);
  const normalizedTimeOfDay = (date - dayOnset) / (1000 * 60 * 60 * 24)
  return {date, normalizedDayOfYear, normalizedTimeOfDay};
}

/**
 * A class that fetches the sprited MNIST dataset and provide data as
 * tf.Tensors.
 */
export class JenaWeatherData {
  constructor() {}

  /**
   * Load and preprocess data.
   *
   * This method first tries to load the data from `FAST_JENA_WEATHER_CSV_PATH`
   * (a relative path) and, if that fails, will try to load it from a remote
   * URL (`JENA_WEATHER_CSV_PATH`).
   */
  async load() {
    let csvData;
    try {
      csvData = await (await fetch(FAST_JENA_WEATHER_CSV_PATH)).text();
      console.log('Loaded data from fast path');
    } catch (err) {
      csvData = await (await fetch(JENA_WEATHER_CSV_PATH)).text();
      console.log('Loaded data from remote path');
    }

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
    // Account for the fact that the first column of the csv file is date-time.
    this.tempCol--;

    this.dateTime = [];
    this.data = [];  // Unnormalized data.
    // Day of the year data, normalized between 0 and 1.
    this.normalizedDayOfYear = [];
    // Time of the day, normalized between 0 and 1.
    this.normalizedTimeOfDay = [];
    for (let i = 1; i < csvLines.length; ++i) {
      const line = csvLines[i].trim();
      if (line.length === 0) {
        continue;
      }
      const items = line.split(',');
      const parsed = parseDateTime(items[0]);
      const newDateTime = parsed.date;
      if (this.dateTime.length > 0 &&
          newDateTime.getTime() <=
              this.dateTime[this.dateTime.length - 1].getTime()) {
      }

      this.dateTime.push(newDateTime);
      this.data.push(items.slice(1).map(x => +x));
      this.normalizedDayOfYear.push(parsed.normalizedDayOfYear);
      this.normalizedTimeOfDay.push(parsed.normalizedTimeOfDay);
    }
    this.numRows = this.data.length;
    this.numColumns = this.data[0].length;
    this.numColumnsExcludingTarget = this.data[0].length - 1;
    console.log(
        `this.numColumnsExcludingTarget = ${this.numColumnsExcludingTarget}`);

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

    // Cache normalized values.
    this.normalizedData = [];
    for (let i = 0; i < this.numRows; ++i) {
      const row = [];
      for (let j = 0; j < this.numColumns; ++j) {
        row.push((this.data[i][j] - this.means[j]) / this.stddevs[j]);
      }
      this.normalizedData.push(row);
    }
  }

  getDataColumnNames() {
    return this.dataColumnNames;
  }

  getTime(index) {
    return this.dateTime[index];
  }

  /**
   * Get the mean and standard deviation of a data column.
   *
   *
   */
  getMeanAndStddev(dataColumnName) {
    if (this.means == null || this.stddevs == null) {
      throw new Error('means and stddevs have not been calculated yet.');
    }

    const index = this.getDataColumnNames().indexOf(dataColumnName);
    if (index === -1) {
      throw new Error(`Invalid data column name: ${dataColumnName}`);
    }
    return {
      mean: this.means[index], stddev: this.stddevs[index]
    }
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
      let value = normalize ? this.normalizedData[i][columnIndex] :
                              this.data[i][columnIndex];
      if (includeTime) {
        value = {x: this.dateTime[i].getTime(), y: value};
      }
      out.push(value);
    }
    return out;
  }

  /**
   * Get a data iterator function.
   *
   * @param {boolean} shuffle Whether the data is to be shuffled. If `false`,
   *   the examples generated by repeated calling of the returned iterator
   *   function will scan through range specified by `minIndex` and `maxIndex`
   *   (or the entire range of the CSV file if those are not specified) in a
   *   sequential fashion. If `true`, the examples generated by the returned
   *   iterator function will start from random rows.
   * @param {number} lookBack Number of look-back time steps. This is how many
   *   steps to look back back when making a prediction. Typical value: 10 days
   *   (i.e., 6 * 24 * 10 = 1440).
   * @param {number} delay Number of time steps from the last time point in the
   *   input features to the time of prediction. Typical value: 1 day (i.e.,
   *   6 * 24 = 144).
   * @param {number} batchSize Batch size.
   * @param {number} step Number of steps between consecutive time points in the
   *   input features. This is a downsampling factor for the input features.
   *   Typical value: 1 hour (i.e., 6).
   * @param {number} minIndex Optional minimum index to draw from the original
   *   data set. Together with `maxIndex`, this can be used to reserve a chunk
   *   of the original data for validation or evaluation.
   * @param {number} maxIndex Optional maximum index to draw from the original
   *   data set. Together with `minIndex`, this can be used to reserve a chunk
   *   of the original data for validation or evaluation.
   * @param {boolean} normalize Whether the iterator function will return
   *   normalized data.
   * @param {boolean} includeDateTime Include the date-time features, including
   *   normalized day-of-the-year and normalized time-of-the-day.
   * @return {Function} An iterator Function, which returns a batch of features
   *   and targets when invoked. The features and targets are arranged in a
   *   length-2 array, in the said order.
   *   The features are represented as a float32-type `tf.Tensor` of shape
   *     `[batchSize, Math.floor(lookBack / step), featureLength]`
   *   The targets are represented as a float32-type `tf.Tensor` of shape
   *     `[batchSize, 1]`.
   */
  getIteratorFn(
      shuffle, lookBack, delay, batchSize, step, minIndex, maxIndex, normalize,
      includeDateTime) {
    let i = minIndex + lookBack;
    const lookBackSlices = Math.floor(lookBack / step);

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

      const featureLength =
          includeDateTime ? this.numColumns + 2 : this.numColumns;
      const samples = tf.buffer([numExamples, lookBackSlices, featureLength]);
      const targets = tf.buffer([numExamples, 1]);
      for (let j = 0; j < numExamples; ++j) {
        const row = rows[j];
        let exampleRow = 0;
        for (let r = row - lookBack; r < row; r += step) {
          let exampleCol = 0;
          for (let n = 0; n < featureLength; ++n) {
            let value;
            if (n < this.numColumns) {
              value = normalize ? this.normalizedData[r][n] : this.data[r][n];
            } else if (n === this.numColumns) {
              // Normalized day-of-the-year feature.
              value = this.normalizedDayOfYear[r];
            } else {
              // Normalized time-of-the-day feature.
              value = this.normalizedTimeOfDay[r];
            }
            samples.set(value, j, exampleRow, exampleCol++);
          }

          const value = normalize ?
              this.normalizedData[r + delay][this.tempCol] :
              this.data[r + delay][this.tempCol];
          targets.set(value, j, 0);
          exampleRow++;
        }
      }
      return {
        value: [samples.toTensor(), targets.toTensor()],
        done: false
      };  // TODO(cais): Return done = true when done.
    }

    return iteratorFn.bind(this);
  }
}
