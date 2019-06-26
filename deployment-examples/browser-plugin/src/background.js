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

// import * as tf from '@tensorflow/tfjs';

// import {IMAGENET_CLASSES} from './imagenet_classes';

// const MOBILENET_MODEL_PATH =
//     // tslint:disable-next-line:max-line-length
//     'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json';

// const IMAGE_SIZE = 224;
// const TOPK_PREDICTIONS = 10;

// let mobilenet;

/**
 * Loads the model and calls predict on some garbage data to make subsequet
 * calls faster.
 */
/*const loadMobilenetDemo = async () => {
  status('Loading model...');
  mobilenet = await tf.loadLayersModel(MOBILENET_MODEL_PATH);
  mobilenet.predict(tf.zeros([1, IMAGE_SIZE, IMAGE_SIZE, 3])).dispose();
  status('Model warmed up....');
};
*/
/**
 * Given an image element, makes a prediction through mobilenet returning the
 * probabilities of the top K classes.
 */
/*async function predict(imgElement) {
  status('Predicting...');
  // The first start time includes the time it takes to extract the image
  // from the HTML and preprocess it, in additon to the predict() call.
  const startTime1 = performance.now();
  // The second start time excludes the extraction and preprocessing and
  // includes only the predict() call.
  let startTime2;
  const logits = tf.tidy(() => {
    // tf.browser.fromPixels() returns a Tensor from an image element.
    const img = tf.browser.fromPixels(imgElement).toFloat();
    const offset = tf.scalar(127.5);
    // Normalize the image from [0, 255] to [-1, 1].
    const normalized = img.sub(offset).div(offset);
    // Reshape to a single-element batch so we can pass it to predict.
    const batched = normalized.reshape([1, IMAGE_SIZE, IMAGE_SIZE, 3]);
    startTime2 = performance.now();
    // Make a prediction through mobilenet.
    return mobilenet.predict(batched);
  });
  // Convert logits to probabilities and class names.
  const classes = await getTopKClasses(logits, TOPK_PREDICTIONS);
  const totalTime1 = performance.now() - startTime1;
  const totalTime2 = performance.now() - startTime2;
  status(`Done in ${Math.floor(totalTime1)} ms ` +
      `(not including preprocessing: ${Math.floor(totalTime2)} ms)`);
  // Show the classes in the DOM.
  showResults(imgElement, classes);
}
*/

function handleImageURL(url) {
  console.log('IM HANDLING IM HANDLING');
  // TODO: Create a div element in the main page's DOM
  // TODO: Load the image into that div
  // TODO: Execute `predict` on the image element.
};

/**
 * Illustrates the results (currently logs to console).
 * @param {*} imgElement
 * @param {*} classes
 */
/*function showResults(imgElement, classes) {
  console.log("I'm showResults!");
  for (let i = 0; i < TOPK_PREDICTIONS; i++) {
    console.log(IMAGENET_CLASSES[i]);
  }
}
*/
const status = (x) => console.log(x);

//loadMobilenetDemo();

/**
 * @fileoverview Description of this file.
 */
chrome.runtime.onInstalled.addListener(function() {
  chrome.storage.sync.set({color: '#3aa757'}, function() {
    console.log("The color is green.");
  });
  chrome.declarativeContent.onPageChanged.removeRules(undefined, function() {
    chrome.declarativeContent.onPageChanged.addRules([{
      conditions: [new chrome.declarativeContent.PageStateMatcher({
        pageUrl: {hostEquals: 'google.com'},
      })],
      actions: [new chrome.declarativeContent.ShowPageAction()]
    }]);
  });
});

/*chrome.runtime.onMessage.addListener(
  function(message, callback) {
    if (message == "predict") {
    }
  }
); */

// Registers a context menu option for the predict handler.
chrome.contextMenus.create({
  title: "Use URL of image somehow",
  contexts:["image"],
  onclick: function(info) {
    console.log(info);
    handleImageURL(info.srcUrl);
    chrome.runtime.sendMessage({
      msg: "something_completed",
      data: {
        subject: "stansSubject",
        content: "stansContent"
      }
    })
  }  
});

