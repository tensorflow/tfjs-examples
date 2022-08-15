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

import * as tf from '@tensorflow/tfjs';
import * as tfVis from '@tensorflow/tfjs-vis';

import * as loader from './loader';
import * as ui from './ui';
import * as util from './util';

const HOSTED_URLS = {
  model:
      'https://storage.googleapis.com/tfjs-models/tfjs/mnist_transfer_cnn_v1/model.json',
  train:
      'https://storage.googleapis.com/tfjs-models/tfjs/mnist_transfer_cnn_v1/gte5.train.json',
  test:
      'https://storage.googleapis.com/tfjs-models/tfjs/mnist_transfer_cnn_v1/gte5.test.json'
};

const LOCAL_URLS = {
  model: 'http://localhost:1235/resources/model.json',
  train: 'http://localhost:1235/resources/gte5.train.json',
  test: 'http://localhost:1235/resources/gte5.test.json'
};

class MnistTransferCNNPredictor {
  /**
   * Initializes the MNIST Transfer CNN demo.
   */
  async init(urls) {
    this.urls = urls;
    this.model = await loader.loadHostedPretrainedModel(urls.model);

    // Print model summary right after model is loaded.
    this.model.summary();
    tfVis.show.modelSummary(
        {name: 'Model Summary', tab: 'Model Info'}, this.model);

    this.imageSize = this.model.layers[0].batchInputShape[1];
    this.numClasses = 5;

    await this.loadRetrainData();
    this.prepTestExamples();
    return this;
  }

  async loadRetrainData() {
    ui.status('Loading data for transfer learning...');
    this.gte5TrainData =
        await loader.loadHostedData(this.urls.train, this.numClasses);
    this.gte5TestData =
        await loader.loadHostedData(this.urls.test, this.numClasses);
    ui.status('Done loading data for transfer learning.');
  }

  prepTestExamples() {
    // Some hard-coded MNIST image examples for interactive testing.
    const testExamples = {};
    const digitCounts = {5: 0, 6: 0, 7: 0, 8: 0, 9: 0};
    const examplesPerDigit = 10;
    // Enter one example of each of 5, 6, 7, 8, 9 in `testExamples`.
    for (let i = this.gte5TestData.data.length - 1; i >= 0; --i) {
      const datum = this.gte5TestData.data[i];
      const digit = datum.y + 5;
      if (digitCounts[digit] >= examplesPerDigit) {
        continue;
      }
      digitCounts[digit]++;
      const key = String(digit) + '_' + String(digitCounts[digit]);
      testExamples[key] = [];
      for (const row of datum.x) {
        testExamples[key] = testExamples[key].concat(row);
      }
      if (Object.keys(testExamples).length >= 5 * examplesPerDigit) {
        break;
      }
    }

    this.testExamples = testExamples;
  }

  // Perform prediction on the input image using the loaded model.
  predict(imageText) {
    tf.tidy(() => {
      try {
        const image = util.textToImageArray(imageText, this.imageSize);
        const predictOut = this.model.predict(image);
        const winner = predictOut.argMax(1);

        ui.setPredictResults(predictOut.dataSync(), winner.dataSync()[0] + 5);
      } catch (e) {
        ui.setPredictError(e.message);
      }
    });
  }

  // Perform retraining on the loaded model.
  async retrainModel() {
    ui.status(
        'Please wait and do NOT click anything while the model retrains...',
        'blue');

    const trainingMode = ui.getTrainingMode();
    if (trainingMode === 'freeze-feature-layers') {
      console.log('Freezing feature layers of the model.');
      for (let i = 0; i < 7; ++i) {
        this.model.layers[i].trainable = false;
      }
    } else if (trainingMode === 'reinitialize-weights') {
      // Make a model with the same topology as before, but with re-initialized
      // weight values.
      const returnString = false;
      this.model = await tf.models.modelFromJSON({
        modelTopology: this.model.toJSON(null, returnString)
      });
    }
    this.model.compile({
      loss: 'categoricalCrossentropy',
      optimizer: tf.train.adam(0.01),
      metrics: ['acc'],
    });

    // Print model summary again after compile(). You should see a number
    // of the model's weights have become non-trainable.
    this.model.summary();

    const batchSize = 128;
    const epochs = ui.getEpochs();

    const surfaceInfo = {name: trainingMode, tab: 'Transfer Learning'};
    console.log('Calling model.fit()');
    await this.model.fit(this.gte5TrainData.x, this.gte5TrainData.y, {
      batchSize: batchSize,
      epochs: epochs,
      validationData: [this.gte5TestData.x, this.gte5TestData.y],
      callbacks: [
        ui.getProgressBarCallbackConfig(epochs),
        tfVis.show.fitCallbacks(surfaceInfo, ['val_loss', 'val_acc'], {
          zoomToFit: true,
          zoomToFitAccuracy: true,
          height: 200,
          callbacks: ['onEpochEnd'],
        }),
      ]
    });
    console.log('DONE Calling model.fit()');
  }
}

/**
 * Loads the pretrained model and metadata, and registers the predict
 * and retrain functions with the UI.
 */
async function setupMnistTransferCNN() {
  if (await loader.urlExists(HOSTED_URLS.model)) {
    ui.status('Model available: ' + HOSTED_URLS.model);
    const button = document.getElementById('load-pretrained-remote');
    button.addEventListener('click', async () => {
      const predictor = await new MnistTransferCNNPredictor().init(HOSTED_URLS);
      ui.prepUI(
          x => predictor.predict(x), () => predictor.retrainModel(),
          predictor.testExamples, predictor.imageSize);
    });
    button.style.display = 'inline-block';
  }

  if (await loader.urlExists(LOCAL_URLS.model)) {
    ui.status('Model available: ' + LOCAL_URLS.model);
    const button = document.getElementById('load-pretrained-local');
    button.addEventListener('click', async () => {
      const predictor = await new MnistTransferCNNPredictor().init(LOCAL_URLS);
      ui.prepUI(
          x => predictor.predict(x), () => predictor.retrainModel(),
          predictor.testExamples, predictor.imageSize);
    });
    button.style.display = 'inline-block';
  }

  ui.status('Standing by. Please load pretrained model first.');
}

setupMnistTransferCNN();
